#version 120

/*
	gbuffers_terrain.fsh
	
	Terrain geometry pass - processes solid blocks.
	
	Responsibilities:
	- Sample albedo texture and apply vertex color/lighting
	- Detect material properties (smoothness, metallic, f0) from texture
	- Output to G-buffer for deferred lighting in composite pass
	
	PBRlite material detection inspired by Photon Shaders (Sixthsurge)
*/

#define LIGHTBOOST 1.0  // global lighting multiplier

uniform sampler2D texture;
uniform sampler2D lightmap;
uniform float rainStrength;

varying vec4 color;
varying vec4 texcoord;
varying vec4 lmcoord;
varying vec2 rawLightmap; // Raw light values: x = block light, y = sky light (0-1)
varying vec4 bloommask;

const int GL_LINEAR = 9729;
const int GL_EXP = 2048;
const float PI = 3.14159265359;

uniform int fogMode;


// ============================================================================
//  MATH UTILITIES
// ============================================================================

float sqr(float x) { return x * x; }
float cube(float x) { return x * x * x; }
float pow4(float x) { float x2 = x * x; return x2 * x2; }
float pow5(float x) { return pow4(x) * x; }
float pow8(float x) { float x2 = x * x; float x4 = x2 * x2; return x4 * x4; }

// Attempt at an unclamped lerp
float linear_step(float edge0, float edge1, float x) {
	return clamp((x - edge0) / (edge1 - edge0), 0.0, 1.0);
}


// ============================================================================
//  COLOR UTILITIES
// ============================================================================

// Convert RGB to HSL (Hue, Saturation, Lightness)
vec3 rgb_to_hsl(vec3 c) {
	const vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
	vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
	vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));
	float d = q.x - min(q.w, q.y);
	float e = 1e-6;
	return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// Isolate a specific hue range (0-360 degrees)
// Returns strength based on how close the color is to targetHue
float isolate_hue(vec3 hsl, float targetHue, float width) {
	float hueDist = abs(hsl.x * 360.0 - targetHue);
	hueDist = min(hueDist, 360.0 - hueDist);
	return (1.0 - linear_step(0.0, width, hueDist)) * hsl.y;
}

// Perceptual luminance (Rec. 709 coefficients)
float pbrLuma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

// Color saturation (HSV-style: chroma / max)
float pbrSaturation(vec3 c) {
	float mx = max(c.r, max(c.g, c.b));
	float mn = min(c.r, min(c.g, c.b));
	return (mx > 0.0) ? (mx - mn) / mx : 0.0;
}


// ============================================================================
//  TEXTURE ANALYSIS
// ============================================================================
// These functions analyze texture properties to estimate material smoothness.
// High-detail textures (like ores) are treated as rougher surfaces.

// Snap UV to texel center to prevent shimmer at distance
vec2 snapToTexelCenter(vec2 uv) {
	vec2 texelSize = fwidth(uv);
	return (floor(uv / texelSize) + 0.5) * texelSize;
}

// Sample brightness difference from a neighbor texel (for detail detection)
float pbrGetDif(float lOriginal, vec2 offsetCoord) {
	const float normalThreshold = 0.05;  // ignore small differences
	const float normalClamp = 0.2;       // clamp extreme differences
	
	float lNearby = length(texture2D(texture, offsetCoord).rgb);
	float dif = lOriginal - lNearby;
	
	if (dif > 0.0) dif = max(dif - normalThreshold, 0.0);
	else           dif = min(dif + normalThreshold, 0.0);
	
	return clamp(dif, -normalClamp, normalClamp);
}

// Measure local texture detail (high = rough surface like ore, low = smooth like metal)
float pbrTextureDetail(vec2 uv, vec3 centerColor) {
	vec2 snappedUV = snapToTexelCenter(uv);
	vec2 texelStep = fwidth(uv);
	float lOriginal = length(centerColor);
	
	// sample at snapped positions offset by full texels
	float difU = pbrGetDif(lOriginal, snappedUV + vec2(0.0, texelStep.y));
	float difD = pbrGetDif(lOriginal, snappedUV - vec2(0.0, texelStep.y));
	float difR = pbrGetDif(lOriginal, snappedUV + vec2(texelStep.x, 0.0));
	float difL = pbrGetDif(lOriginal, snappedUV - vec2(texelStep.x, 0.0));
	
	float detail = abs(difU) + abs(difD) + abs(difR) + abs(difL);
	
	// fade detail at distance to prevent shimmer
	float texelScreenSize = length(texelStep) * 512.0;
	float distanceFade = smoothstep(0.5, 2.0, texelScreenSize);
	detail *= distanceFade;
	
	return detail;
}


// ============================================================================
//  MATERIAL DETECTION (PBRlite)
// ============================================================================
// Automatically detect material properties from texture color/pattern.
// Uses heuristics based on color (hue, saturation, brightness) and texture detail.
// Returns vec4(roughness, f0, highlightMult, metallic)
//   - roughness: 0=mirror, 1=diffuse
//   - f0: Fresnel reflectance at normal incidence (0.04=plastic, 1.0=metal)
//   - highlightMult: specular intensity multiplier
//   - metallic: 0=dielectric, 1=metal

vec4 pbrLiteMaterial(vec4 texSample, vec2 uv) {
	vec3 c = texSample.rgb;
	vec3 hsl = rgb_to_hsl(c);
	float luma = pbrLuma(c);
	float sat = pbrSaturation(c);
	float detail = pbrTextureDetail(uv, c);
	
	// Pre-computed color channel powers for material matching
	float colorG = c.g;
	float colorG2 = colorG * colorG;
	float colorG4 = colorG2 * colorG2;
	float colorR2 = c.r * c.r;
	float colorR4 = colorR2 * colorR2;
	
	// Material defaults (generic block)
	float smoothnessG = 0.33 * smoothstep(0.2, 0.6, hsl.z);
	float f0 = 0.04;          // dielectric default
	float highlightMult = 1.0;
	float metallic = 0.0;
	
	// --- Color Classification ---
	float grayness = 1.0 - sat;
	float isGray = smoothstep(0.0, 0.15, grayness);
	
	// Hue detection (degrees)
	float isYellow = isolate_hue(hsl, 50.0, 30.0);
	float isOrange = isolate_hue(hsl, 30.0, 20.0);
	float isBlue = isolate_hue(hsl, 210.0, 40.0);
	float isGreen = isolate_hue(hsl, 120.0, 40.0);
	float isCyan = isolate_hue(hsl, 180.0, 30.0);
	float isRed = isolate_hue(hsl, 0.0, 25.0);
	
	// Brightness-based detection
	float isWhite = smoothstep(0.75, 0.92, luma) * smoothstep(0.12, 0.0, sat);  // quartz, snow
	float isSandy = smoothstep(0.5, 0.8, luma) * smoothstep(0.08, 0.25, sat)    // sand, sandstone
	              * smoothstep(0.0, 0.15, c.r - c.b);
	
	// --- METALS ---
	
	// Iron: gray, medium brightness, low saturation
	float ironLike = isGray * smoothstep(0.3, 0.6, luma) * smoothstep(0.1, 0.0, sat);
	smoothnessG += ironLike * colorR4 * 1.0;
	highlightMult += ironLike * colorR4 * 1.5;
	f0 = mix(f0, 0.78, ironLike);
	metallic += ironLike * 0.65;
	
	// Gold: yellow hue, high reflectance
	smoothnessG += isYellow * min(max(colorG, 0.8) - colorG4 * 0.5, 1.0) * 0.8;
	highlightMult += isYellow * 2.0 * max(colorG4, 0.2);
	f0 = mix(f0, 1.0, isYellow * 0.9);
	metallic += isYellow * 0.75;
	
	// Copper: orange hue
	smoothnessG += isOrange * colorR2 * 0.6;
	highlightMult += isOrange * 1.4;
	f0 = mix(f0, 0.95, isOrange * 0.7);
	metallic += isOrange * 0.55;
	
	// --- ORES ---
	// Detected by high texture detail (ore veins create local contrast)
	float isOre = smoothstep(0.08, 0.20, detail);
	
	// Iron ore
	float ironOre = isOre * isGray * smoothstep(0.2, 0.5, luma);
	smoothnessG += ironOre * 0.4;
	highlightMult += ironOre * 0.8;
	f0 = mix(f0, 0.5, ironOre * 0.5);
	metallic += ironOre * 0.3;
	
	// gold ore
	float goldOre = isOre * isYellow;
	smoothnessG += goldOre * 0.5;
	highlightMult += goldOre * 1.2;
	f0 = mix(f0, 0.9, goldOre * 0.6);
	metallic += goldOre * 0.4;
	
	// redstone ore
	float redstoneOre = isOre * isRed * smoothstep(0.3, 0.5, sat);
	smoothnessG += redstoneOre * 0.35;
	highlightMult += redstoneOre * 0.6;
	f0 = mix(f0, 0.15, redstoneOre);
	
	// lapis ore
	float lapisOre = isOre * isBlue;
	smoothnessG += lapisOre * 0.45;
	highlightMult += lapisOre * 0.9;
	f0 = mix(f0, 0.2, lapisOre * 0.5);
	
	// coal ore
	float coalOre = isOre * smoothstep(0.3, 0.1, luma) * isGray;
	smoothnessG += coalOre * 0.25;
	highlightMult += coalOre * 0.4;
	
	// --- STONES ---
	// Gray blocks that aren't metal or ore
	float stoneLike = isGray * (1.0 - ironLike) * (1.0 - isWhite) * (1.0 - isOre * 0.7);
	smoothnessG += stoneLike * min(colorG4 * 1.5, 0.7);
	highlightMult += stoneLike * 0.35;
	
	// --- GEMS & CRYSTALS ---
	
	// Diamond/Lapis: blue, high reflectance
	smoothnessG += isBlue * 0.65;
	highlightMult += isBlue * 1.2;
	f0 = mix(f0, 0.25, isBlue * 0.7);
	
	// Emerald: saturated green
	float emeraldLike = isGreen * smoothstep(0.4, 0.6, sat);
	smoothnessG += emeraldLike * 0.6;
	highlightMult += emeraldLike * 1.0;
	f0 = mix(f0, 0.22, emeraldLike * 0.6);
	
	// Prismarine: cyan tint
	smoothnessG += isCyan * 0.45;
	highlightMult += isCyan * 0.7;
	f0 = mix(f0, 0.12, isCyan * 0.5);
	
	// --- OTHER BLOCKS ---
	
	// Bright blocks (quartz, calcite, snow): slightly reflective
	smoothnessG += isWhite * colorG2 * 0.55;
	highlightMult += isWhite * (colorG2 * 0.5 + 0.4);
	f0 = mix(f0, 0.08, isWhite * 0.5);
	
	// Organic materials (grass, leaves, wood): saturated colors = rough
	float organic = sat * (1.0 - isYellow) * (1.0 - isBlue) * (1.0 - isGreen) * (1.0 - metallic);
	smoothnessG -= organic * 0.35;
	
	// --- CONTRAST ADAPTATION ---
	// High-detail textures get specular boost, uniform surfaces get dampened
	float contrastFactor = smoothstep(0.02, 0.15, detail);
	
	// High-contrast blocks (ores, detailed stone): boost specular
	smoothnessG += contrastFactor * 0.2;
	highlightMult += contrastFactor * 0.4;
	
	// Low-contrast bright blocks (wool, concrete): dampen specular
	float uniformBright = (1.0 - contrastFactor) * smoothstep(0.5, 0.8, luma);
	smoothnessG -= uniformBright * 0.4;
	highlightMult -= uniformBright * 0.3;
	
	// Sandy blocks: extra rough
	float isSandyUniform = isSandy * (1.0 - contrastFactor);
	smoothnessG -= isSandyUniform * 0.3;
	
	// Dark blocks shouldn't be shiny (prevents coal/obsidian glare)
	smoothnessG *= smoothstep(0.05, 0.25, luma);
	highlightMult *= smoothstep(0.1, 0.3, luma);
	
	// --- FINAL OUTPUT ---
	smoothnessG = clamp(smoothnessG, 0.0, 1.0);
	
	// Convert smoothness to roughness: roughness = (1 - smoothness)²
	float roughness = sqr(1.0 - smoothnessG);
	roughness = max(roughness, 0.04);  // minimum to prevent singularities
	
	f0 = clamp(f0, 0.02, 1.0);
	highlightMult = max(highlightMult, 0.5);
	metallic = clamp(metallic, 0.0, 1.0);
	
	return vec4(roughness, f0, highlightMult, metallic);
}


// ============================================================================
//  MAIN
// ============================================================================

void main() {
	vec4 baseTex = texture2D(texture, texcoord.st);
	vec4 lm = texture2D(lightmap, lmcoord.st);
	
	// Detect material properties from texture
	vec4 matProps = pbrLiteMaterial(baseTex, texcoord.st);
	float roughness = matProps.x;
	float f0 = matProps.y;
	float highlightMult = matProps.z;
	float metallic = matProps.w;
	
	// Convert roughness back to smoothness for G-buffer storage
	float smoothness = 1.0 - sqrt(roughness);
	
	// --- G-Buffer Outputs ---
	// FragData[0]: Albedo (RGB) with vanilla lighting
	gl_FragData[0] = baseTex * lm * LIGHTBOOST * color;
	
	// FragData[1]: Depth (for shadow mapping)
	gl_FragData[1] = vec4(vec3(gl_FragCoord.z), 1.0);
	
	// FragData[2] (gnormal.r): lighting luminance proxy (legacy-safe)
	// Macula legacy pipeline note: `color` is often already lit/tinted, so dividing by
	// (baseTex * color) collapses to ~1. Instead, divide the *lit output* by the raw
	// texel luminance to approximate the lighting factor.
	vec3 litColor = (baseTex.rgb * lm.rgb) * (color.rgb * LIGHTBOOST);
	float litLum = dot(litColor, vec3(0.2126, 0.7152, 0.0722));
	float texLum = dot(baseTex.rgb, vec3(0.2126, 0.7152, 0.0722));
	float lightProxy = clamp(litLum / max(texLum, 0.04), 0.0, 1.0);
	gl_FragData[2] = vec4(lightProxy, 0.0, 0.0, 1.0);
	
	// FragData[4] (gaux1): Material properties packed into RGBA
	//   R: f0 (0-0.49, scaled to avoid conflict with hand shader)
	//   G: metallic (0-0.49, scaled to avoid conflict with water mask)
	//   B: land mask (1.0 = solid geometry, 0.0 = sky)
	//   A: smoothness (0-1)
	float f0Packed = clamp(f0 * 0.49, 0.0, 0.49);
	float metallicPacked = clamp(metallic * 0.49, 0.0, 0.49);
	gl_FragData[4] = vec4(f0Packed, metallicPacked, 1.0, smoothness);
	
	// fog applied in composite.fsh
}