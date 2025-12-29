#version 120

/*
	IdeaLegacy Shaders by icee

	Credits:
	- Daxnitro (original GLSL Shader Mod)
	- Sonic Ether (base shader as reference, reimplemented)
	- jbritain/Glimmer Shaders (water reflections)
	- EminGT/Complementary Reimagined & Sixthsurge/Photon Shaders (PBRlite)
	- ICGI by yours truly :>

	Sources & Licenses:
	- Complementary Reimagined: https://www.complementary.dev/ (See LICENSE_COMPLEMENTARY.txt)
	- Photon Shaders: https://github.com/sixthsurge/photon (See LICENSE_PHOTON.txt)
	- Glimmer Shaders: https://github.com/jbritain/glimmer-shaders (See LICENSE_GLIMMER.txt)
*/

/*
	NOTES: This shader was tested on the Macula shader mod for Babric, Minecraft Beta 1.7.3.
	It will likely work with other mods, like the OG GLSL Shader mod or Optifine, 
	but I can't guarantee that. Some workarounds are used to increase compatibility, and
	add functionality on top of the (frankly broken) legacy shader pipeline.
*/

/*
	KNOWN BUGS:
	- When testing on Linux with an Nvidia GPU, screen overlays,
	like Fire and Water caused spontaneous flickering.
		- Using Zink instead of native OpenGL fixed this.

	- Rain looks Bad™. 
		- Wontfix. Can't mask rain from the sky cleanly.

	- In dark areas, the transition is poor from light-to-dark,
	having areas of negative contrast and looking washed out.
		- This is an artifact of the lightening pass used to raise
		complete shadow like caves away from actual 0. 
		I can't figure out how to fix this without compromising contrast.

	- When your camera approaches the edge of a block, the scene darkens.
		- Use Zink if this darkening looks glitchy like overlays. 
		No idea why the darkening happens though. sorry
	
	- Player models and hand viewport appears behind the sky in certain circumstances.
		- Again, no clue.
	
	- Specular highlights in the Nether appear to be tied to the camera
		- Cantfix. Sky updating breaks in the nether and the sun position stops working.
		I also don't want to turn off PBRlite in the nether.
	
*/


// post-processing toggles (comment to disable)
#define FXAA
#define VIGNETTE
#define CROSSPROCESS
#define HIGHDESATURATE

// Debug: show gnormal.r (legacy-safe rawLightmap.y replacement) as grayscale
// Uncomment to enable.
// #define DEBUG_LIGHT_PROXY

// Optional: brighten the debug view for easier inspection
// const float DEBUG_LIGHT_PROXY_GAIN = 1.0;

// color grading
#define VIGNETTE_STRENGTH 1.1
#define BRIGHTMULT 1.00      // 1.0 = default, higher = brighter
#define DARKMULT 0.00        // 0.0 = normal, higher = darker darks
#define COLOR_BOOST 0.25     // 0.0 = normal, higher = more saturated
#define GAMMA 0.82           // 1.0 = default, lower = brighter


uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D composite;
uniform sampler2D gnormal;
uniform sampler2D gaux1;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferPreviousProjection;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousModelView;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

uniform vec3 sunPosition;

uniform int worldTime;
uniform float aspectRatio;
uniform float near;
uniform float far;
uniform float viewWidth;
uniform float viewHeight;
uniform float rainStrength;

uniform int fogMode;
const int GL_LINEAR = 9729;
const int GL_EXP = 2048;

varying vec4 texcoord;

float linearizeDepth(vec2 coord);
vec4 getTimeOfDay(float wtime);


// ============================================================================
//  HDR / TONEMAPPING
// ============================================================================

#define HDR_RGBM_RANGE 24.0    // HDR encoding range for RGBM format
#define TONEMAP_EXPOSURE 1.9   // exposure multiplier before tonemapping

vec3 decodeRGBM(vec4 rgmb) {
	return rgmb.rgb;
}

// Uncharted 2 filmic tonemapping (attempt to look more natural via s-curve)
// Creates a natural shoulder rolloff for highlights and slight lift in shadows
vec3 tonemap_uc2_partial(vec3 x) {
	const float a = 0.15;  // shoulder strength
	const float b = 0.50;  // linear strength
	const float c = 0.10;  // linear angle
	const float d = 0.20;  // toe strength
	const float e = 0.02;  // toe numerator
	const float f = 0.30;  // toe denominator
	return ((x * (a * x + (c * b)) + (d * e)) / (x * (a * x + b) + d * f)) - e / f;
}

vec3 tonemapUncharted2(vec3 x) {
	x = max(x, vec3(0.0));
	x *= TONEMAP_EXPOSURE;
	const float exposure_bias = 2.0;
	const vec3 w = vec3(11.2);
	vec3 curr = tonemap_uc2_partial(x * exposure_bias);
	vec3 white_scale = vec3(1.0) / tonemap_uc2_partial(w);
	return clamp(curr * white_scale, 0.0, 1.0);
}

// Legacy alias - kept for compatibility with older code paths
vec3 tonemapACES(vec3 x) {
	return tonemapUncharted2(x);
}

vec3 sampleCompositeHDR(vec2 uv) {
	vec4 enc = texture2D(composite, uv);
	float m = texture2D(gaux1, uv).g;
	return enc.rgb * (m * HDR_RGBM_RANGE);
}

#ifdef ICGI
vec3 icgi_applyHDR(vec2 uv, vec3 hdrColor, bool isNether);
#endif

vec3 sampleCompositeLDR(vec2 uv) {
	return tonemapACES(sampleCompositeHDR(uv));
}

vec4 sampleCompositeLDR4(vec2 uv) {
	return vec4(sampleCompositeLDR(uv), 1.0);
}

// Fast power functions (avoid pow() overhead)
float pow2(float x) { return x * x; }
float pow4(float x) { x *= x; return x * x; }
float pow8(float x) { x *= x; x *= x; return x * x; }


// ============================================================================
//  FOG
// ============================================================================

// Exponential fog transmittance: returns 1.0 (clear) to 0.0 (fully fogged)
float spherical_fog(float view_dist, float start, float density_exp2) {
	return exp2(-density_exp2 * max(view_dist - start, 0.0));
}

// Convert screen UV to world-space ray direction
vec3 getWorldDirFromUV(vec2 uv) {
	vec4 screenRay = gbufferProjectionInverse * vec4(uv.s * 2.0 - 1.0, uv.t * 2.0 - 1.0, 1.0, 1.0);
	screenRay /= max(screenRay.w, 0.0001);
	vec3 viewDir = normalize(screenRay.xyz);
	return normalize((gbufferModelViewInverse * vec4(viewDir, 0.0)).xyz);
}

// Convert planar (Z-buffer) depth to true spherical distance.
// Fixes fog appearing thicker at screen edges where planar depth != actual distance.
float getSphericalDistance(vec2 uv, float planarDepth) {
	vec4 screenRay = gbufferProjectionInverse * vec4(uv.s * 2.0 - 1.0, uv.t * 2.0 - 1.0, 1.0, 1.0);
	screenRay /= max(screenRay.w, 0.0001);
	vec3 viewDir = normalize(screenRay.xyz);
	return planarDepth / max(-viewDir.z, 0.001);
}

// Detect Nether dimension by checking if fog is predominantly red
bool detectNether() {
	vec3 vanillaFog = gl_Fog.color.rgb;
	float fogRedness = vanillaFog.r / (vanillaFog.g + vanillaFog.b + 0.001);
	return fogRedness > 1.5 && vanillaFog.r > 0.1;
}

// Global cache (computed once per frame in main(), used by fog/ICGI functions)
vec4 g_cachedTOD = vec4(0.0);      // time of day weights
bool g_isNether = false;           // dimension detection
vec3 g_cachedFogColor = vec3(0.0); // final fog color after palette + rain

// Compute fog color based on time of day and dimension.
// Returns tonemapped LDR color ready for blending.
vec3 fogPaletteTime(vec4 tod, bool isNether) {
	if (isNether) {
		// Nether: enhance vanilla fog with gamma correction + saturation boost
		vec3 netherFog = gl_Fog.color.rgb;
		netherFog = pow(netherFog, vec3(2.2));  // linear space
		float fogLuma = dot(netherFog, vec3(0.299, 0.587, 0.114));
		netherFog = mix(vec3(fogLuma), netherFog, 1.5);  // saturation boost
		netherFog *= 2.0;
		return netherFog;
	}
	
	// Overworld: blend between time-of-day palette colors
	float tSunrise = tod.x;
	float tNoon    = tod.y;
	float tSunset  = tod.z;
	float tMidnight= tod.w;

	// Time-of-day fog palette (keep in sync with composite.fsh)
	vec3 fogSunrise   = vec3(1.00, 0.55, 0.25);  // warm orange
	vec3 fogNoon      = vec3(0.75, 0.90, 1.00);  // cool blue-white
	vec3 fogSunset    = vec3(0.95, 0.45, 0.75);  // pink-magenta
	vec3 fogMidnight  = vec3(0.00, 0.00, 0.001); // near-black

	vec3 fogTarget = fogSunrise * tSunrise + fogNoon * tNoon + fogSunset * tSunset + fogMidnight * tMidnight;

	// keep some vanilla dimension tint
	vec3 fogColor = mix(gl_Fog.color.rgb, fogTarget, 0.85);

	// saturation boost
	float fogSatBoost = 1.45;
	float fogLumaPreSat = dot(fogColor, vec3(0.299, 0.587, 0.114));
	fogColor = mix(vec3(fogLumaPreSat), fogColor, fogSatBoost);
	fogColor = max(fogColor, vec3(0.0));

	// rain: dim and desaturate
	const float RAIN_SKY_DESAT_MAX = 0.75;
	const float RAIN_SKY_DIM_MULT  = 0.1;
	const float RAIN_SKY_CURVE     = 1.6;
	float rainT = pow(clamp(rainStrength, 0.0, 1.0), RAIN_SKY_CURVE);
	float fogLuma = dot(fogColor, vec3(0.299, 0.587, 0.114));
	vec3 fogGray = vec3(fogLuma);
	fogColor = mix(fogColor, fogGray, rainT * RAIN_SKY_DESAT_MAX);
	fogColor *= mix(1.0, RAIN_SKY_DIM_MULT, rainT);

	// HDR intensity scaling to match sky at horizon
	float curFogLum = dot(fogColor, vec3(0.299, 0.587, 0.114));
	float desiredFogLum = 1.22 * tNoon + 0.85 * (tSunrise + tSunset) + 0.10 * tMidnight;
	desiredFogLum *= mix(1.0, RAIN_SKY_DIM_MULT, rainT);
	float lumScale = desiredFogLum / max(curFogLum, 0.001);
	// relax lower clamp when raining so fog can get darker
	float lumScaleMin = mix(0.5, 0.06, rainT);
	lumScale = clamp(lumScale, lumScaleMin, 4.0);
	fogColor *= lumScale;
	
	// extra rain dimming to match sky
	float extraDim = mix(1.0, 0.12, rainT);
	fogColor *= extraDim;

	// tonemap fog to match sky output
	fogColor = tonemapUncharted2(fogColor);

	return fogColor;
}

// Apply distance fog to scene color.
// Handles both vanilla fog modes and adds "border fog" for smooth horizon blending.
vec3 applyFog(vec2 uv, vec3 color, float landHere) {
	if (landHere <= 0.5) return color;  // skip sky pixels

	float depthLinear = linearizeDepth(uv);
	float sphericalDist = getSphericalDistance(uv, depthLinear);
	
	vec3 fogColor = g_cachedFogColor;

	// Border fog: fade geometry to fog at horizon near the far plane
	vec3 worldDir = getWorldDirFromUV(uv);
	float distFade = smoothstep(far * 0.35, far, sphericalDist);
	float horizon = 1.0 - clamp(abs(worldDir.y), 0.0, 1.0);
	float borderStrength = pow8(horizon) * distFade;
	float borderVis = clamp(1.0 - borderStrength, 0.0, 1.0);
	color = mix(fogColor, color, borderVis);

	// main fog
	float transmittance = 1.0;
	if (fogMode == GL_EXP) {
		// convert e-based density to exp2
		float densityExp2 = max(gl_Fog.density, 0.00001) * 1.442695;
		transmittance = spherical_fog(sphericalDist, max(gl_Fog.start, 0.0), densityExp2);
	} else if (fogMode == GL_LINEAR) {
		float fogAmount = clamp((sphericalDist - gl_Fog.start) * gl_Fog.scale, 0.0, 1.0);
		transmittance = 1.0 - fogAmount;
	} else {
		float fogStart = far * 0.35;
		float fogEnd = far;
		float fogAmount = clamp((sphericalDist - fogStart) / max(fogEnd - fogStart, 0.0001), 0.0, 1.0);
		transmittance = 1.0 - fogAmount;
	}

	vec3 scattering = fogColor * (1.0 - transmittance);
	return color * transmittance + scattering;
}

// Legacy aliases for compatibility
vec3 applyFogHDR(vec2 uv, vec3 hdrColor, float landHere) { return applyFog(uv, hdrColor, landHere); }
vec3 applyFogLDR(vec2 uv, vec3 ldrColor, float landHere) { return applyFog(uv, ldrColor, landHere); }


// ============================================================================
//  TIME OF DAY
// ============================================================================

// Returns blend weights for each time period: vec4(sunrise, noon, sunset, midnight)
// Each component is 0.0-1.0, and they smoothly transition throughout the day.
vec4 getTimeOfDay(float wtime) {
	float tSunrise  = ((clamp(wtime, 23000.0, 24000.0) - 23000.0) / 1000.0) + (1.0 - (clamp(wtime, 0.0, 2000.0)/2000.0));
	float tNoon     = ((clamp(wtime, 0.0, 2000.0)) / 2000.0) - ((clamp(wtime, 10000.0, 12000.0) - 10000.0) / 2000.0);
	float tSunset   = ((clamp(wtime, 10000.0, 12000.0) - 10000.0) / 2000.0) - ((clamp(wtime, 12000.0, 12750.0) - 12000.0) / 750.0);
	float tMidnight = ((clamp(wtime, 12000.0, 12750.0) - 12000.0) / 750.0) - ((clamp(wtime, 23000.0, 24000.0) - 23000.0) / 1000.0);
	return vec4(tSunrise, tNoon, tSunset, tMidnight);
}

// ============================================================================
//  DEPTH UTILITIES
// ============================================================================

// Linearize the hyperbolic Z-buffer value to view-space distance.
// The Z-buffer stores depth non-linearly: z_ndc = (far + near) / (far - near) + 
// (2 * far * near) / ((far - near) * z_view). Solving for z_view gives this formula.
float linearizeDepth(vec2 coord) {
    float z_ndc = texture2D(gdepth, coord).x * 2.0 - 1.0;  // [0,1] -> [-1,1]
    return (2.0 * near * far) / (far + near - z_ndc * (far - near));
}

// Raw depth buffer sample (non-linear, 0=near, 1=far)
float sampleRawDepth(vec2 coord) {
	return texture2D(gdepth, coord).x;
}

// ============================================================================
//  SCENE SAMPLING
// ============================================================================

// Sample scene in LDR (tonemapped, no fog)
vec3 sceneLDR(vec2 uv) {
	return tonemapACES(sampleCompositeHDR(uv));
}

// Sample scene in LDR with fog applied (used by FXAA)
vec3 sceneFoggedLDR(vec2 uv) {
	float landHere = texture2D(gaux1, uv).b;
	vec3 c = sceneLDR(uv);
	return applyFogLDR(uv, c, landHere);
}


// ============================================================================
//  ANTI-ALIASING (FXAA + Outlier Rejection)
// ============================================================================

// Clamp outlier pixels to neighbor luminance range (reduces fireflies/sparkles)
vec3 rejectOutliersLDR(vec2 uv, vec3 centerLDR) {
	float pw = 1.0 / viewWidth;
	float ph = 1.0 / viewHeight;
	
	// sample 4 neighbors
	vec3 baseN = sceneFoggedLDR(uv + vec2(0.0, -ph));
	vec3 baseS = sceneFoggedLDR(uv + vec2(0.0,  ph));
	vec3 baseE = sceneFoggedLDR(uv + vec2( pw, 0.0));
	vec3 baseW = sceneFoggedLDR(uv + vec2(-pw, 0.0));
	
	// luminance comparison
	float lumC = dot(centerLDR, vec3(0.2126, 0.7152, 0.0722));
	float lumN = dot(baseN, vec3(0.2126, 0.7152, 0.0722));
	float lumS = dot(baseS, vec3(0.2126, 0.7152, 0.0722));
	float lumE = dot(baseE, vec3(0.2126, 0.7152, 0.0722));
	float lumW = dot(baseW, vec3(0.2126, 0.7152, 0.0722));
	
	// Clamp to neighbor range (works both for too-bright and too-dark outliers)
	float lumMin = min(min(lumN, lumS), min(lumE, lumW));
	float lumMax = max(max(lumN, lumS), max(lumE, lumW));
	
	// detect and correct outliers
	float outlierBright = smoothstep(0.0, 0.15, lumC - lumMax);
	float outlierDark = smoothstep(0.0, 0.15, lumMin - lumC);
	float correction = max(outlierBright, outlierDark);
	
	vec3 avgNeighbor = (baseN + baseS + baseE + baseW) * 0.25;
	return mix(centerLDR, avgNeighbor, correction);
}

// FXAA (Fast Approximate Anti-Aliasing) applied after fog.
// Detects edges via luminance contrast and blurs along the edge direction.
vec3 fxaaPostFog(vec2 uv, vec3 centerColor) {
	float pw = 1.0 / viewWidth;   // pixel width
	float ph = 1.0 / viewHeight;  // pixel height

	vec3 baseCenter = sceneFoggedLDR(uv);
	vec3 delta = centerColor - baseCenter;

	vec3 luma = vec3(0.299, 0.587, 0.114);
	vec3 rgbNW = sceneFoggedLDR(uv + vec2(-pw, -ph));
	vec3 rgbNE = sceneFoggedLDR(uv + vec2( pw, -ph));
	vec3 rgbSW = sceneFoggedLDR(uv + vec2(-pw,  ph));
	vec3 rgbSE = sceneFoggedLDR(uv + vec2( pw,  ph));

	float lumaNW = dot(rgbNW, luma);
	float lumaNE = dot(rgbNE, luma);
	float lumaSW = dot(rgbSW, luma);
	float lumaSE = dot(rgbSE, luma);
	float lumaM  = dot(baseCenter, luma);

	float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
	float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));

	vec2 dir;
	dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
	dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));

	float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 / 8.0), 1.0 / 128.0);
	float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
	dir = min(vec2(4.0, 4.0), max(vec2(-4.0, -4.0), dir * rcpDirMin)) * vec2(pw, ph);

	vec3 rgbA = 0.5 * (
		sceneFoggedLDR(uv + dir * (1.0 / 3.0 - 0.5)) +
		sceneFoggedLDR(uv + dir * (2.0 / 3.0 - 0.5))
	);
	vec3 rgbB = rgbA * 0.5 + 0.25 * (
		sceneFoggedLDR(uv + dir * (0.0 / 3.0 - 0.5)) +
		sceneFoggedLDR(uv + dir * (3.0 / 3.0 - 0.5))
	);

	float lumaB = dot(rgbB, luma);
	if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
		return clamp(rgbA + delta, 0.0, 1.0);
	}
	return clamp(rgbB + delta, 0.0, 1.0);
}

// Combined outlier rejection + FXAA in a single pass.
// More efficient than separate passes since we reuse neighbor samples.
vec4 outlierRejectFXAA(vec2 uv, bool doFXAA) {
	float pw = 1.0 / viewWidth;
	float ph = 1.0 / viewHeight;
	
	// Sample center pixel + all 8 neighbors
	vec3 rgbM  = sceneLDR(uv);
	vec3 rgbN  = sceneLDR(uv + vec2(0.0, -ph));
	vec3 rgbS  = sceneLDR(uv + vec2(0.0,  ph));
	vec3 rgbE  = sceneLDR(uv + vec2( pw, 0.0));
	vec3 rgbW  = sceneLDR(uv + vec2(-pw, 0.0));
	vec3 rgbNW = sceneLDR(uv + vec2(-pw, -ph));
	vec3 rgbNE = sceneLDR(uv + vec2( pw, -ph));
	vec3 rgbSW = sceneLDR(uv + vec2(-pw,  ph));
	vec3 rgbSE = sceneLDR(uv + vec2( pw,  ph));
	
	// outlier rejection: clamp to neighbor range
	vec3 minN = min(min(min(rgbN, rgbS), min(rgbE, rgbW)), min(min(rgbNE, rgbNW), min(rgbSE, rgbSW)));
	vec3 maxN = max(max(max(rgbN, rgbS), max(rgbE, rgbW)), max(max(rgbNE, rgbNW), max(rgbSE, rgbSW)));
	vec3 clamped = clamp(rgbM, minN, maxN);
	
	vec3 diff = abs(rgbM - clamped);
	float outlierStrength = max(max(diff.r, diff.g), diff.b);
	float correction = smoothstep(0.02, 0.15, outlierStrength);
	vec3 cleanM = mix(rgbM, clamped, correction);
	
	if (!doFXAA) {
		return vec4(cleanM, 1.0);
	}
	
	// FXAA: reuse corner samples
	vec3 luma = vec3(0.299, 0.587, 0.114);
	float lumaNW = dot(rgbNW, luma);
	float lumaNE = dot(rgbNE, luma);
	float lumaSW = dot(rgbSW, luma);
	float lumaSE = dot(rgbSE, luma);
	float lumaM  = dot(cleanM, luma);
	
	float lumaMin = min(lumaM, min(min(lumaNW, lumaNE), min(lumaSW, lumaSE)));
	float lumaMax = max(lumaM, max(max(lumaNW, lumaNE), max(lumaSW, lumaSE)));
	
	vec2 dir;
	dir.x = -((lumaNW + lumaNE) - (lumaSW + lumaSE));
	dir.y =  ((lumaNW + lumaSW) - (lumaNE + lumaSE));
	
	float dirReduce = max((lumaNW + lumaNE + lumaSW + lumaSE) * (0.25 / 8.0), 1.0 / 128.0);
	float rcpDirMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + dirReduce);
	dir = min(vec2(4.0, 4.0), max(vec2(-4.0, -4.0), dir * rcpDirMin)) * vec2(pw, ph);
	
	// FXAA directional samples
	vec3 rgbA = 0.5 * (
		sceneLDR(uv + dir * (1.0 / 3.0 - 0.5)) +
		sceneLDR(uv + dir * (2.0 / 3.0 - 0.5))
	);
	vec3 rgbB = rgbA * 0.5 + 0.25 * (
		sceneLDR(uv + dir * (0.0 / 3.0 - 0.5)) +
		sceneLDR(uv + dir * (3.0 / 3.0 - 0.5))
	);
	
	float lumaB = dot(rgbB, luma);
	if ((lumaB < lumaMin) || (lumaB > lumaMax)) {
		return vec4(rgbA, 1.0);
	}
	return vec4(rgbB, 1.0);
}



// ============================================================================
//  ICGI (Screen-Space Global Illumination)
// ============================================================================
// Approximates indirect lighting by sampling nearby screen colors and using
// them to tint surfaces. Not physically accurate, but adds ambient color
// bleeding that makes scenes feel more cohesive.

#define ICGI

#ifdef ICGI
	// Debug modes: 0=off, 1=show GI only, 2=ambient, 3=unlit, 4=normals
	const int debug_mode = 0;
	
	// GI tuning parameters
	const float gi_strength = 1.0;       // overall effect intensity
	const float gi_saturation = 0.6;     // color saturation of bounced light
	const float gi_contrast = 0.6;       // contrast preservation
	const float OverExposeFactor = 0.5;  // blowout/glow amount
	const float bounce_multiplier = 1.0; // secondary bounce strength
	const float gi_shape = 0.1;          // normal-based offset amount
	
	// Blur radius as percentage of screen height (0.185 ≈ 18.5%)
	const float BlurRadiusPercent = 0.185;
	float BlurRadius = BlurRadiusPercent * viewHeight;
	const float HDR_GI_REFERENCE = 1.5;  // HDR compression reference point
	
	// Nether-specific GI: warm lava-colored ambient
	const vec3  NETHER_GI_COLOR = vec3(1.0, 0.6, 0.5);
	const float NETHER_GI_INTENSITY = 5.0;
	const float NETHER_GI_RADIUS_MULT = 1.8;

	// Reconstruct view-space position from screen UV and depth buffer
	vec3 icgi_getViewPos(vec2 coord) {
		float z = texture2D(gdepth, coord).x;
		vec4 p = gbufferProjectionInverse * vec4(coord.s * 2.0 - 1.0, coord.t * 2.0 - 1.0, 2.0 * z - 1.0, 1.0);
		p /= p.w;
		return p.xyz;
	}

	// Sample scene color for GI gathering.
	// For sky pixels, uses fog color instead of raw sky (prevents sky bleeding artifacts).
	vec3 icgi_sourceHDR(vec2 uv) {
		float landMask = texture2D(gaux1, uv).b;
		vec3 hdr = sampleCompositeHDR(uv);
		
		// sky pixels: use fog color instead of raw composite
		if (landMask < 0.5) {
			vec3 skyFogColor = g_cachedFogColor;
			vec3 worldDir = getWorldDirFromUV(uv);
			float horizonFade = 1.0 - abs(worldDir.y) * 0.3;
			return skyFogColor * horizonFade * 1.5;
		}
		
		return hdr;
	}

	// Soft HDR compression using Reinhard-style curve.
	// Prevents bright areas from dominating the GI average.
	vec3 icgi_softCompressHDR(vec3 hdr) {
		return hdr / (1.0 + hdr / HDR_GI_REFERENCE);
	}
	
	// Inverse of soft compression (expand back to HDR range)
	vec3 icgi_softExpandHDR(vec3 compressed) {
		vec3 safeC = min(compressed, vec3(HDR_GI_REFERENCE * 0.99));
		return safeC * HDR_GI_REFERENCE / max(HDR_GI_REFERENCE - safeC, vec3(0.01));
	}

	vec3 icgi_tapPoisson(vec2 baseUV, mat2 rot, vec2 dir, float radiusPx, vec2 invTex, out float weight) {
		vec2 dd = rot * dir;
		float dist = length(dd);
		
		// distance-based falloff
		weight = 1.0 / (1.0 + dist * dist * 4.0);
		
		vec2 suv = clamp(baseUV + dd * radiusPx * invTex, vec2(0.015), vec2(0.985));
		vec3 hdr = icgi_sourceHDR(suv);
		
		return icgi_softCompressHDR(hdr) * weight;
	}

	// Large-radius Poisson disk blur for gathering ambient light.
	// Uses 12 samples with random rotation to hide banding artifacts.
	vec3 icgi_bigBlur(vec2 uv, float radiusPx) {
		vec2 texSize = vec2(viewWidth, viewHeight);
		vec2 invTex = vec2(1.0 / viewWidth, 1.0 / viewHeight);
		vec2 baseUV = clamp(uv, vec2(0.015), vec2(0.985));

		float rand = fract(sin(dot(baseUV * texSize, vec2(12.9898, 78.233))) * 43758.5453);
		float angle = rand * 6.2831853;
		mat2 rot = mat2(cos(angle), -sin(angle), sin(angle), cos(angle));

		// 12 poisson disk samples
		vec2 d0  = vec2(-0.326212, -0.405810);
		vec2 d1  = vec2(-0.840144, -0.073580);
		vec2 d2  = vec2(-0.695914,  0.457137);
		vec2 d3  = vec2( 0.962340, -0.194983);
		vec2 d4  = vec2( 0.473434, -0.480026);
		vec2 d5  = vec2( 0.519456,  0.767022);
		vec2 d6  = vec2( 0.185461, -0.893124);
		vec2 d7  = vec2( 0.896420,  0.412458);
		vec2 d8  = vec2(-0.321940, -0.932615);
		vec2 d9  = vec2(-0.791559, -0.597710);
		vec2 d10 = vec2(-0.558520,  0.324530);
		vec2 d11 = vec2( 0.772630, -0.627420);

		vec3 sum = vec3(0.0);
		float totalWeight = 0.0;
		float w;
		
		sum += icgi_tapPoisson(baseUV, rot, d0,  radiusPx, invTex, w); totalWeight += w;
		sum += icgi_tapPoisson(baseUV, rot, d1,  radiusPx, invTex, w); totalWeight += w;
		sum += icgi_tapPoisson(baseUV, rot, d2,  radiusPx, invTex, w); totalWeight += w;
		sum += icgi_tapPoisson(baseUV, rot, d3,  radiusPx, invTex, w); totalWeight += w;
		sum += icgi_tapPoisson(baseUV, rot, d4,  radiusPx, invTex, w); totalWeight += w;
		sum += icgi_tapPoisson(baseUV, rot, d5,  radiusPx, invTex, w); totalWeight += w;
		sum += icgi_tapPoisson(baseUV, rot, d6,  radiusPx, invTex, w); totalWeight += w;
		sum += icgi_tapPoisson(baseUV, rot, d7,  radiusPx, invTex, w); totalWeight += w;
		sum += icgi_tapPoisson(baseUV, rot, d8,  radiusPx, invTex, w); totalWeight += w;
		sum += icgi_tapPoisson(baseUV, rot, d9,  radiusPx, invTex, w); totalWeight += w;
		sum += icgi_tapPoisson(baseUV, rot, d10, radiusPx, invTex, w); totalWeight += w;
		sum += icgi_tapPoisson(baseUV, rot, d11, radiusPx, invTex, w); totalWeight += w;

		vec3 compressed = sum / max(totalWeight, 0.001);
		return icgi_softExpandHDR(compressed);
	}

	// Main ICGI function: apply fake global illumination to HDR scene color.
	// Gathers nearby colors, estimates surface normal from depth, and blends
	// a color-tinted ambient term into the scene.
	vec3 icgi_applyHDR(vec2 uv, vec3 hdrColor, bool isNether) {
		float effectiveStrength = gi_strength;
		float effectiveSaturation = gi_saturation;
		
		vec3 preToneColor = hdrColor;
		float lum = dot(preToneColor, vec3(0.3333));
		float fade = smoothstep(0.0, 0.05, lum);

		vec3 bounceColor = preToneColor * normalize(max(preToneColor, vec3(1e-5)));

		vec3 bounceArea = preToneColor;

		// depth-derived normal
		float pw = 1.0 / viewWidth;
		float ph = 1.0 / viewHeight;
		vec3 posC = icgi_getViewPos(uv);
		vec3 posR = icgi_getViewPos(uv + vec2(pw, 0.0));
		vec3 posU = icgi_getViewPos(uv + vec2(0.0, ph));
		vec3 dx = posR - posC;
		vec3 dy = posU - posC;
		vec3 n = normalize(cross(dx, dy));
		if (dot(n, posC) > 0.0) n = -n;
		vec2 giOffset = n.xy * gi_shape;
		vec2 normalVec = (gi_shape > 0.0) ? (giOffset / gi_shape) : vec2(0.0);

		vec2 giBaseUV = clamp(uv + giOffset, vec2(0.015), vec2(0.985));
		vec3 giSample = icgi_bigBlur(giBaseUV, BlurRadius) + vec3(0.001);

		// nether: force GI to lava color
		if (isNether) {
			float giBrightRaw = dot(giSample, vec3(0.2126, 0.7152, 0.0722));
			giSample = NETHER_GI_COLOR * giBrightRaw * NETHER_GI_INTENSITY;
		}

		// HDR-safe brightness calculation
		float giLuma = dot(giSample, vec3(0.2126, 0.7152, 0.0722));
		float giBrightness = giLuma * min(giLuma, 2.0);  // Soft clamp at 2.0 to prevent runaway

		float lightFactor = giBrightness + 0.005;
		vec3 unlitColor2 = preToneColor / max(lightFactor, 1e-5);
		bounceColor = mix(bounceColor, unlitColor2 * max(vec3(0.0), 2.0 * bounceArea - preToneColor), bounce_multiplier);

		float smoothBrightness = dot(preToneColor, vec3(0.2126, 0.7152, 0.0722));
		smoothBrightness = smoothBrightness * min(smoothBrightness, 2.0);
		float ambientTerm = mix(giBrightness, smoothBrightness, 0.5);
		vec3 unlitColor = preToneColor / max(ambientTerm, 1e-5);

		vec3 grayGI = vec3(length(giSample) / 1.41421);
		float satDen = length(preToneColor * (1.0 - effectiveSaturation) + giSample * effectiveSaturation);
		float satNum = length(giSample * effectiveSaturation);
		float satMix = (satDen > 1e-6) ? (satNum / satDen) : 0.0;
		vec3 desatGI = mix(grayGI, giSample, satMix);
		vec3 giBounce = unlitColor * 2.0 * desatGI;

		float denom = length(bounceArea + giSample);
		float currContrast = max(sqrt(2.0 * length(preToneColor) / max(denom, 1e-6)), 0.75);

		vec3 midColor = (giSample + preToneColor) * 0.5;
		vec3 minRGB = min(midColor, giBounce);
		float lenMid = length(minRGB);
		float lenOrig = length(preToneColor);
		float adjustedLength = clamp(lenMid, lenOrig, 1.5 * lenOrig);

		vec3 originalColor = preToneColor;
		vec3 mainColor = normalize(max(giBounce, vec3(1e-6))) * adjustedLength;
		mainColor *= mix(1.0, currContrast, gi_contrast);

		vec3 intermediateColor = mainColor;
		vec3 originalGI = giSample;

		vec3 giSampleScaled = giSample * giBrightness;
		vec3 blowoutColor = giSampleScaled * giSampleScaled;
		float blowoutIntensity = clamp(dot(blowoutColor, vec3(0.3333)), 0.0, 1.0);
		giSampleScaled = mix(giSampleScaled, originalGI, 0.5);

		mainColor *= (giSampleScaled + 0.75);
		mainColor = mix(mainColor, intermediateColor, 0.5);

		vec3 blownColor = mix(mainColor, giSampleScaled, blowoutIntensity);
		mainColor = mix(mainColor, blownColor, OverExposeFactor * 0.15);
		
		mainColor = mix(originalColor, mainColor, effectiveStrength);

		if (debug_mode == 1) mainColor = giSample;
		if (debug_mode == 2) mainColor = vec3(ambientTerm);
		if (debug_mode == 3) mainColor = unlitColor;
		if (debug_mode == 4) mainColor = vec3(((-normalVec) + 1.0) * 0.5, 1.0);

		return mix(originalColor, mainColor, fade);  // Return HDR result
	}
#endif


// ============================================================================
//  MAIN
// ============================================================================

void main() {
	#ifdef DEBUG_LIGHT_PROXY
		float proxy = texture2D(gnormal, texcoord.st).r;
		float land = texture2D(gaux1, texcoord.st).b;
		float dbg = clamp(proxy, 0.0, 1.0);
		// More-informative encoding so "all white" becomes diagnosable:
		//   R = dbg (linear)
		//   G = sqrt(dbg) (compress highlights)
		//   B = 1 - dbg (invert)
		vec3 encoded = vec3(dbg, sqrt(dbg), 1.0 - dbg);
		// Visualize on geometry; show sky as blue so cleared gnormal is obvious
		vec3 outRgb = mix(vec3(0.0, 0.0, 1.0), encoded, step(0.5, land));
		gl_FragColor = vec4(outRgb, 1.0);
		return;
	#endif

	// Sample G-buffer data once
	vec4 auxSample = texture2D(gaux1, texcoord.st);
	float gcolorAlpha = texture2D(gcolor, texcoord.st).a;
	float rawDepth = texture2D(gdepth, texcoord.st).x;
	float landHere = auxSample.b;  // 1.0 = land, 0.0 = sky
	float aHere = gcolorAlpha;

	// Initialize global caches (used by fog and ICGI functions)
	g_cachedTOD = getTimeOfDay(float(worldTime));
	g_isNether = detectNether();
	g_cachedFogColor = fogPaletteTime(g_cachedTOD, g_isNether);

	// --- HDR Processing ---
	vec3 hdrColor = sampleCompositeHDR(texcoord.st);
	
	// Apply ICGI (only to solid geometry, not sky or translucent)
	#ifdef ICGI
		if (landHere > 0.5 && aHere >= 0.999 && rawDepth < 0.9999) {
			hdrColor = icgi_applyHDR(texcoord.st, hdrColor, g_isNether);
		}
	#endif
	
	// --- LDR Processing ---
	vec4 color = vec4(tonemapACES(hdrColor), 1.0);
	
	color.rgb = applyFogLDR(texcoord.st, color.rgb, landHere);
	color.rgb = rejectOutliersLDR(texcoord.st, color.rgb);

	// --- Vignette ---
	// Uses squared distance for efficiency, dual-smoothstep for natural falloff curve
	#ifdef VIGNETTE
		// Aspect-corrected UV coordinates centered at screen middle
		vec2 vignetteUV = texcoord.st - 0.5;
		vignetteUV.x *= aspectRatio;  // correct for non-square aspect ratios
		
		// Squared radial distance (cheaper than length(), smooth gradient)
		float vignetteDistSq = dot(vignetteUV, vignetteUV);
		
		// Dual-smoothstep creates natural photographic vignette curve:
		// - Inner region (< 0.15 radius²) stays fully bright
		// - Middle region fades gradually
		// - Corners reach maximum darkening
		float vignetteInner = 1.0 - smoothstep(0.15, 0.55, vignetteDistSq);
		float vignetteOuter = 1.0 - smoothstep(0.4, 1.2, vignetteDistSq);
		float vignetteFactor = vignetteInner * 0.6 + vignetteOuter * 0.4;
		
		// Apply darkening: corners reach ~25% darker at max strength
		float vignetteDarken = mix(1.0 - 0.25 * VIGNETTE_STRENGTH, 1.0, vignetteFactor);
		color.rgb *= clamp(vignetteDarken, 0.0, 1.0);
	#endif

	color.rgb *= BRIGHTMULT;

	// --- Cross-Processing ---
	// Emulates C-41 cross-process look: cool shadows, warm highlights, lifted blacks
	#ifdef CROSSPROCESS
		// Pre-gain and gentle lift (closer to beta look, alternative formula)
		color.rgb = color.rgb * (BRIGHTMULT + 0.015) + vec3(0.025);
		
		// Per-channel gamma shaping driven by inverse brightness (not the same as beta)
		// inv approximates db = -color + 1.4 in beta, but with milder response
		vec3 inv = 1.35 - color.rgb;
		vec3 baseGamma = vec3(0.66, 0.68, 0.72);
		vec3 gammaMix = mix(baseGamma, inv, 0.55);
		
		// Small per-channel pre-scale and offset before gamma
		vec3 pre = color.rgb * vec3(0.965, 0.97, 0.98) - vec3(0.003);
		pre = clamp(pre, 0.0, 1.0);
		color.rgb = pow(pre, gammaMix);
		
		// Subtle tonal split to retain cool shadows / warm highlights
		float lumaCross = dot(color.rgb, vec3(0.299, 0.587, 0.114));
		vec3 highlightTint = vec3(1.015, 1.0, 0.985);
		vec3 shadowTint = vec3(0.985, 0.995, 1.015);
		vec3 crossTint = mix(shadowTint, highlightTint, smoothstep(0.30, 0.80, lumaCross));
		color.rgb *= crossTint;
	#endif

	// --- Saturation Boost ---
	// Boosts muted colors more than already-saturated colors to prevent clipping
	
	// Compute luminance using Rec.709 coefficients
	float luma = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
	
	// Calculate current saturation level (max channel deviation from luma)
	float maxChannel = max(max(color.r, color.g), color.b);
	float minChannel = min(min(color.r, color.g), color.b);
	float currentSat = (maxChannel - minChannel) / max(maxChannel, 0.001);
	
	// Vibrance: reduce boost for already-saturated colors to prevent oversaturation
	float vibranceFactor = 1.0 - currentSat * 0.5;  // less boost when already saturated
	float effectiveBoost = COLOR_BOOST * vibranceFactor;
	
	// Apply luminance-preserving saturation adjustment
	color.rgb = mix(vec3(luma), color.rgb, 1.0 + effectiveBoost);
	color.rgb = max(color.rgb, vec3(0.0));  // prevent negative values

	// --- Highlight Desaturation ---
	// Emulates film stock behavior where highlights gracefully desaturate toward white
	#ifdef HIGHDESATURATE
		// Compute luminance for desaturation target (Rec.601 for perceptual accuracy)
		float highlightLuma = dot(color.rgb, vec3(0.299, 0.587, 0.114));
		
		// Two-stage desaturation: gradual onset + accelerating rolloff
		// Stage 1: begins subtly at mid-tones
		float desatOnset = smoothstep(0.45, 0.8, highlightLuma);
		// Stage 2: accelerates in bright highlights
		float desatRolloff = smoothstep(0.7, 1.0, highlightLuma);
		// Combined: smooth natural curve
		float desatAmount = desatOnset * 0.35 + desatRolloff * 0.4;
		
		// Blend toward luminance (grayscale) in highlights
		color.rgb = mix(color.rgb, vec3(highlightLuma), desatAmount);
		
		// --- Final Color Adjustments ---
		// Gamma correction: <1.0 brightens midtones, >1.0 darkens
		color.rgb = pow(max(color.rgb, vec3(0.0)), vec3(GAMMA));
		
		// Black level adjustment: lifts or crushes shadows
		// DARKMULT > 0 crushes blacks, DARKMULT < 0 would lift them
		color.rgb = color.rgb * (1.0 + DARKMULT) - DARKMULT;
		color.rgb = max(color.rgb, vec3(0.0));  // clamp negative values
	#endif

	// --- Anti-Aliasing ---
	// FXAA applied last so it smooths the final composited image
	#ifdef FXAA
		color.rgb = fxaaPostFog(texcoord.st, color.rgb);
	#endif

	gl_FragColor = color;
}
