#version 120


// ============================================================================
//  SHADOW CONFIGURATION
// ============================================================================

#define SHADOW_HQ              // high quality shadow filtering
#define SHADOW_SOFTNESS 4.0    // shadow blur radius (higher = softer edges)
#define SHADOW_INTENSITY 1.0   // 1.0 = default, 2.0 = pitch black, 0.0 = no shadows
#define SHADOW_RANGE 512.0     // max shadow render distance in blocks
#define SHADOW_PCF             // enable PCF shadow filtering

#define SHADOW_DEPTH_SCALE 255.95
#define SHADOW_BIAS 0.05       // bias to prevent shadow acne

// Shadow offset in world/voxel space (blocks) to correct misalignment
const vec3 SHADOW_OFFSET = vec3(0.29166, 0.0, 0.0); //fix for shadow offset

/* SHADOWRES:2048 */           // shadow map resolution
/* SHADOWHPL:100.0 */          // shadow half-plane distance

// ============================================================================
//  PBR SPECULAR SETTINGS
// ============================================================================

const float PBR_SMOOTHNESS_MULT = 1.0;    // global smoothness multiplier
const float PBR_ENV_STRENGTH = 0.5;       // environment/sky reflection intensity
const float PBR_SUN_STRENGTH = 1.0;       // sun specular highlight intensity
const float PBR_MOON_STRENGTH = 0.5;      // moon specular highlight intensity


// ============================================================================
//  WEATHER SETTINGS
// ============================================================================

const float RAIN_SKY_DESAT_MAX = 0.75;    // max desaturation during rain (0-1)
const float RAIN_SKY_DIM_MULT  = 0.1;     // brightness multiplier at full rain
const float RAIN_SKY_CURVE     = 1.6;     // curve power (>1 = faster ramp)


// ============================================================================
//  CELESTIAL BODY SETTINGS
// ============================================================================

// Angular radius for area-light specular (higher = softer, more diffuse highlights)
const float SUN_ANGULAR_RADIUS = 0.3;
const float MOON_ANGULAR_RADIUS = 0.3;


// ============================================================================
//  EMISSIVE DETECTION SETTINGS
// ============================================================================

#define EMISSIVE_DETECTION          // enable emissive light source detection
//#define EMISSIVE_DEBUG            // visualize detected emissives as white
const float EMISSIVE_BLUR_RADIUS = 0.20;   // blur radius as fraction of screen height
const float EMISSIVE_THRESHOLD = 0.12;     // minimum difference to consider emissive (lower = more sensitive)
const float EMISSIVE_BOOST = 1.8;          // HDR intensity boost for emissives (1.0 = no boost)
const float EMISSIVE_SATURATION = 1.15;    // saturation boost for emissive glow (1.0 = no change)
const float EMISSIVE_DARKNESS_BIAS = 2.0;  // extra boost in dark scenes (higher = more contrast)


uniform sampler2D gcolor;
uniform sampler2D gdepth;
uniform sampler2D gnormal;
uniform sampler2D composite;
uniform sampler2D shadow;
uniform sampler2D gaux1;

varying vec4 texcoord;
varying vec4 lmcoord;

uniform int worldTime;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferProjection;
uniform mat4 gbufferModelView;
uniform mat4 gbufferModelViewInverse;
uniform mat4 shadowProjection;
uniform mat4 shadowModelView;

uniform float near;
uniform float far;
uniform float viewWidth;
uniform float viewHeight;
uniform float rainStrength;

uniform vec3 cameraPosition;

uniform vec3 sunPosition;
uniform vec3 moonPosition;

uniform float frameTimeCounter;

const int GL_LINEAR = 9729;
const int GL_EXP = 2048;

const float PI = 3.141592653589793;


// ============================================================================
//  MATH UTILITIES
// ============================================================================

// Fast power functions (avoid pow() overhead)
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
//  FRESNEL
// ============================================================================

// Schlick's approximation for Fresnel reflectance
// f0 = reflectance at normal incidence (0.04 for dielectrics, higher for metals)
vec3 fresnel_schlick(float cos_theta, vec3 f0) {
	float x = 1.0 - cos_theta;
	float x5 = pow5(x);
	return f0 + (1.0 - f0) * x5;
}

float fresnel_schlick_scalar(float cos_theta, float f0) {
	float x = 1.0 - cos_theta;
	float x5 = pow5(x);
	return f0 + (1.0 - f0) * x5;
}

uniform int fogMode;


// ============================================================================
//  HDR ENCODING
// ============================================================================

// RGBM encoding for storing HDR values in RGB8 framebuffer.
// The M (multiplier) channel stores intensity, allowing values up to HDR_RGBM_RANGE.
#define HDR_RGBM_RANGE 24.0

vec4 encodeRGBM(vec3 hdr) {
	hdr = max(hdr, vec3(0.0));
	float maxRGB = max(max(hdr.r, hdr.g), hdr.b);
	float m = clamp(maxRGB / HDR_RGBM_RANGE, 0.0, 1.0);
	m = ceil(m * 255.0) / 255.0;
	vec3 rgb = (m > 0.0) ? (hdr / (m * HDR_RGBM_RANGE)) : vec3(0.0);
	rgb = clamp(rgb, 0.0, 1.0);
	return vec4(rgb, m);
}


// ============================================================================
//  EMISSIVE DETECTION
// ============================================================================
// Detects emissive light sources by comparing local lighting buffer values
// against a blurred average. Areas significantly brighter than their surroundings
// are flagged as emissive and receive an HDR boost for improved contrast.

#ifdef EMISSIVE_DETECTION

// Blur the lighting buffer (gnormal.r) using a 12-sample Poisson disk.
// Returns the average lighting value in the surrounding area.
float blurLightingBuffer(vec2 uv, float radiusPx) {
	vec2 invTex = vec2(1.0 / viewWidth, 1.0 / viewHeight);
	vec2 baseUV = clamp(uv, vec2(0.01), vec2(0.99));
	
	// Random rotation to hide banding artifacts
	float rand = fract(sin(dot(baseUV * vec2(viewWidth, viewHeight), vec2(12.9898, 78.233))) * 43758.5453);
	float angle = rand * 6.2831853;
	float ca = cos(angle);
	float sa = sin(angle);
	mat2 rot = mat2(ca, -sa, sa, ca);
	
	// 12-sample Poisson disk offsets
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
	
	float sum = 0.0;
	float totalWeight = 0.0;
	
	// Sample with distance-based falloff
	#define LIGHT_TAP(dir) { \
		vec2 dd = rot * dir; \
		float dist = length(dd); \
		float w = 1.0 / (1.0 + dist * dist * 2.0); \
		vec2 suv = clamp(baseUV + dd * radiusPx * invTex, vec2(0.01), vec2(0.99)); \
		sum += texture2D(gnormal, suv).r * w; \
		totalWeight += w; \
	}
	
	LIGHT_TAP(d0);  LIGHT_TAP(d1);  LIGHT_TAP(d2);  LIGHT_TAP(d3);
	LIGHT_TAP(d4);  LIGHT_TAP(d5);  LIGHT_TAP(d6);  LIGHT_TAP(d7);
	LIGHT_TAP(d8);  LIGHT_TAP(d9);  LIGHT_TAP(d10); LIGHT_TAP(d11);
	
	#undef LIGHT_TAP
	
	return sum / max(totalWeight, 0.001);
}

// Compute emissive strength by comparing local vs blurred lighting.
// Returns a value 0-1 indicating how "emissive" this pixel appears.
float computeEmissiveStrength(vec2 uv, float localLight, float blurredLight, float sceneLuma) {
	// Difference between local lighting and surrounding average
	float lightDiff = localLight - blurredLight;
	
	// Only consider positive differences (brighter than surroundings)
	if (lightDiff < EMISSIVE_THRESHOLD) return 0.0;
	
	// Normalize difference into 0-1 range with soft curve
	float emissiveRaw = smoothstep(EMISSIVE_THRESHOLD, EMISSIVE_THRESHOLD + 0.4, lightDiff);
	
	// Boost effect in darker scenes for better contrast
	// In bright scenes, emissives matter less; in dark scenes they pop
	float darknessFactor = 1.0 - smoothstep(0.1, 0.5, blurredLight);
	emissiveRaw *= (1.0 + darknessFactor * EMISSIVE_DARKNESS_BIAS);
	
	// Also consider the absolute brightness - very bright pixels are more likely emissive
	float brightnessFactor = smoothstep(0.3, 0.8, localLight);
	emissiveRaw *= (0.5 + brightnessFactor * 0.5);
	
	return clamp(emissiveRaw, 0.0, 1.0);
}

// Apply emissive boost to scene color
vec3 applyEmissiveBoost(vec3 color, float emissiveStrength, float localLight) {
	if (emissiveStrength < 0.01) return color;
	
	// Compute boost amount
	float boost = 1.0 + emissiveStrength * (EMISSIVE_BOOST - 1.0);
	
	// Boost saturation for emissive glow (warm lights look more saturated)
	float luma = dot(color, vec3(0.2126, 0.7152, 0.0722));
	vec3 saturated = mix(vec3(luma), color, EMISSIVE_SATURATION);
	
	// Blend between original and saturated based on emissive strength
	vec3 boosted = mix(color, saturated, emissiveStrength * 0.5);
	
	// Apply brightness boost
	boosted *= boost;
	
	return boosted;
}

#endif // EMISSIVE_DETECTION


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

// Reconstruct view-space position from screen UV and depth buffer
vec3 getViewPos(vec2 coord) {
	float z = texture2D(gdepth, coord).x;
	vec4 p = gbufferProjectionInverse * vec4(coord.s * 2.0 - 1.0, coord.t * 2.0 - 1.0, 2.0 * z - 1.0, 1.0);
	p /= p.w;
	return p.xyz;
}


// ============================================================================
//  NORMAL RECONSTRUCTION
// ============================================================================

// Reconstruct surface normal from depth buffer with edge detection.
// Returns vec4(normal.xyz, confidence) where confidence is low at depth edges
// to prevent specular artifacts at object silhouettes.
#define EDGE_DEPTH_THRESHOLD 0.015

vec4 edgeAwareNormal(vec2 uv, float pw, float ph, float filterStrength, vec3 centerPos, vec3 precomputedPosR, vec3 precomputedPosU) {
	// reuse pre-computed positions if available
	vec3 posR = (length(precomputedPosR) > 0.001) ? precomputedPosR : getViewPos(uv + vec2(pw, 0.0));
	vec3 posU = (length(precomputedPosU) > 0.001) ? precomputedPosU : getViewPos(uv + vec2(0.0, ph));
	vec3 posL = getViewPos(uv + vec2(-pw, 0.0));
	vec3 posD = getViewPos(uv + vec2(0.0, -ph));

	float depth0 = -centerPos.z;
	float depthThreshold = EDGE_DEPTH_THRESHOLD * depth0;
	
	float depthR = -posR.z;
	float depthL = -posL.z;
	float depthU = -posU.z;
	float depthD = -posD.z;
	
	// detect depth discontinuities
	float maxDiff = max(max(abs(depth0 - depthR), abs(depth0 - depthL)),
	                    max(abs(depth0 - depthU), abs(depth0 - depthD)));
	float edgeFactor = smoothstep(depthThreshold * 0.5, depthThreshold * 2.0, maxDiff);
	
	vec3 N = normalize(cross(posR - centerPos, posU - centerPos));
	if (dot(N, centerPos) > 0.0) N = -N;
	
	// at edges, return normal with low confidence
	if (edgeFactor > 0.5) {
		vec3 viewN = normalize(-centerPos);
		N = normalize(mix(N, viewN, edgeFactor * 0.3));
		return vec4(N, 1.0 - edgeFactor * 0.85);
	}
	
	// not at edge: apply lightweight bilateral smoothing
	if (filterStrength > 0.01) {
		// approximate neighbor normals using already-fetched positions
		vec3 NR = normalize(cross(posR - centerPos, posU - centerPos));
		vec3 NL = normalize(cross(centerPos - posL, posU - centerPos));  // Left-facing approx
		vec3 NU = normalize(cross(posR - centerPos, posU - centerPos));  // Up-facing approx
		vec3 ND = normalize(cross(posR - centerPos, centerPos - posD));  // Down-facing approx
		
		// Fix winding
		if (dot(NR, posR) > 0.0) NR = -NR;
		if (dot(NL, posL) > 0.0) NL = -NL;
		if (dot(NU, posU) > 0.0) NU = -NU;
		if (dot(ND, posD) > 0.0) ND = -ND;
		
		// weighted average
		float wR = max(dot(N, NR), 0.0);
		float wL = max(dot(N, NL), 0.0);
		float wU = max(dot(N, NU), 0.0);
		float wD = max(dot(N, ND), 0.0);
		
		float blendAmt = filterStrength * 0.4 * (1.0 - edgeFactor);
		vec3 avgN = N + (NR * wR + NL * wL + NU * wU + ND * wD) * blendAmt;
		N = normalize(avgN);
	}
	
	float confidence = 1.0 - edgeFactor * 0.7;
	return vec4(N, confidence);
}


// ============================================================================
//  BLENDING UTILITIES
// ============================================================================

// Soft light blend mode (Photoshop-style)
// Useful for shadow application without crushing blacks
float softLight(float base, float blend) {
	base = clamp(base, 0.0, 1.0);
	blend = clamp(blend, 0.0, 1.0);
	if (blend < 0.5) {
		return 2.0 * base * blend + base * base * (1.0 - 2.0 * blend);
	}
	return sqrt(base) * (2.0 * blend - 1.0) + 2.0 * base * (1.0 - blend);
}

vec3 softLight(vec3 base, float blend) {
	return vec3(
		softLight(base.r, blend),
		softLight(base.g, blend),
		softLight(base.b, blend)
	);
}


// ============================================================================
//  TIME OF DAY
// ============================================================================

// Returns blend weights for each time period: vec4(sunrise, noon, sunset, midnight)
// Based on Minecraft's worldTime (0-24000 ticks per day)
vec4 getTimeOfDay(float wtime) {
	float tSunrise  = ((clamp(wtime, 23000.0, 24000.0) - 23000.0) / 1000.0) + (1.0 - (clamp(wtime, 0.0, 2000.0)/2000.0));
	float tNoon     = ((clamp(wtime, 0.0, 2000.0)) / 2000.0) - ((clamp(wtime, 10000.0, 12000.0) - 10000.0) / 2000.0);
	float tSunset   = ((clamp(wtime, 10000.0, 12000.0) - 10000.0) / 2000.0) - ((clamp(wtime, 12000.0, 12750.0) - 12000.0) / 750.0);
	float tMidnight = ((clamp(wtime, 12000.0, 12750.0) - 12000.0) / 750.0) - ((clamp(wtime, 23000.0, 24000.0) - 23000.0) / 1000.0);
	return vec4(tSunrise, tNoon, tSunset, tMidnight);
}


// ============================================================================
//  PBR SPECULAR (GGX BRDF)
// ============================================================================

// GGX/Trowbridge-Reitz normal distribution function
float D_GGX(float NoH, float a2) {
	float denom = NoH * NoH * (a2 - 1.0) + 1.0;
	return a2 / (PI * denom * denom);
}

// Schlick-GGX geometry term (single direction)
float G_SchlickGGX(float NdotV, float k) {
	return NdotV / (NdotV * (1.0 - k) + k);
}

// Smith geometry term (combines view and light directions)
float G_Smith(float NoV, float NoL, float roughness) {
	float r1 = roughness + 1.0;
	float k = (r1 * r1) * 0.125; // (r+1)^2 / 8 for direct lighting
	return G_SchlickGGX(NoV, k) * G_SchlickGGX(NoL, k);
}

vec3 fresnelSchlick(float cosTheta, vec3 F0) {
	float x = 1.0 - cosTheta;
	float x2 = x * x;
	float x5 = x2 * x2 * x;
	return F0 + (1.0 - F0) * x5;
}

float fresnelSchlickF(float cosTheta, float F0) {
	float x = 1.0 - cosTheta;
	float x2 = x * x;
	float x5 = x2 * x2 * x;
	return F0 + (1.0 - F0) * x5;
}

// Full GGX specular BRDF (Cook-Torrance microfacet model)
// N: surface normal, V: view direction, L: light direction
// smoothness: 0=rough, 1=mirror, F0: reflectance at normal incidence
vec3 GGX_Specular(vec3 N, vec3 V, vec3 L, float smoothness, vec3 F0) {
	// Convert smoothness to roughness (Photon-style formula)
	float roughness = sqr(1.0 - smoothness);
	roughness = max(roughness, 0.04);  // prevent divide-by-zero
	float alpha = roughness * roughness;
	
	vec3 H = normalize(L + V);
	float NoH = max(dot(N, H), 0.0);
	float NoV = max(dot(N, V), 0.001);
	float NoL = max(dot(N, L), 0.0);
	float VoH = max(dot(V, H), 0.0);
	float LoH = max(dot(L, H), 0.0);
	
	// GGX/Trowbridge-Reitz Distribution
	float alpha2 = alpha * alpha;
	float denom = NoH * NoH * (alpha2 - 1.0) + 1.0;
	float D = alpha2 / (PI * denom * denom);
	
	// Fresnel-Schlick with material-specific f0
	vec3 F = fresnelSchlick(VoH, F0);
	
	// Smith geometry term (height-correlated)
	float k = alpha * 0.5;
	float G_V = NoV / (NoV * (1.0 - k) + k);
	float G_L = NoL / (NoL * (1.0 - k) + k);
	float G = G_V * G_L;
	
	// Full Cook-Torrance specular BRDF
	vec3 specular = vec3(D * G / (4.0 * NoV * NoL + 0.001));
	specular *= F * NoL;
	specular = max(specular, vec3(0.0));
	
	// Tonemap to prevent overly bright highlights
	specular = specular / (vec3(0.125) * specular + vec3(1.0));
	
	return specular;
}

// Area light GGX specular - creates softer highlights for sun/moon
// Uses representative point method: increases effective roughness based on light size
// angularRadius: apparent angular size of light source (higher = softer highlight)
vec3 GGX_Specular_AreaLight(vec3 N, vec3 V, vec3 L, float smoothness, vec3 F0, float angularRadius) {
	float roughness = sqr(1.0 - smoothness);
	roughness = max(roughness, 0.04);
	
	// Combine surface roughness with light source angular size
	// The light source size effectively adds to the surface roughness
	float areaRoughness = sqrt(roughness * roughness + angularRadius * angularRadius);
	float alpha = areaRoughness * areaRoughness;
	
	vec3 H = normalize(L + V);
	float NoH = max(dot(N, H), 0.0);
	float NoV = max(dot(N, V), 0.001);
	float NoL = max(dot(N, L), 0.0);
	float VoH = max(dot(V, H), 0.0);
	
	float alpha2 = alpha * alpha;
	float denom = NoH * NoH * (alpha2 - 1.0) + 1.0;
	float D = alpha2 / (PI * denom * denom);
	
	vec3 F = fresnelSchlick(VoH, F0);
	
	// geometry term (original roughness)
	float origAlpha = roughness * roughness;
	float k = origAlpha * 0.5;
	float G_V = NoV / (NoV * (1.0 - k) + k);
	float G_L = NoL / (NoL * (1.0 - k) + k);
	float G = G_V * G_L;
	
	vec3 specular = vec3(D * G / (4.0 * NoV * NoL + 0.001));
	specular *= F * NoL;
	specular = max(specular, vec3(0.0));
	
	// soft tonemap for area glow
	specular = specular / (vec3(0.08) * specular + vec3(1.0));
	
	return specular;
}

vec3 GGX_Specular(vec3 N, vec3 V, vec3 L, float smoothness) {
	return GGX_Specular(N, V, L, smoothness, vec3(0.04));
}


// ============================================================================
//  WATER WAVES & REFLECTIONS
// ============================================================================
// Procedural wave simulation using sum of Gerstner-like waves.
// Based on techniques from Glimmer Shaders (jbritain).

#define DRAG_MULT 0.2              // wave interaction/choppiness
#define WAVE_E 0.1                 // wave energy
#define WAVE_DEPTH 0.3             // wave displacement depth
#define WATER_PARALLAX_SAMPLES 8   // parallax refinement steps

float clamp01(float x) { return clamp(x, 0.0, 1.0); }

// Animation time source (prefers frameTimeCounter, falls back to worldTime)
float waveTime() {
	float t = frameTimeCounter;
	return (t > 0.0001) ? t : (float(worldTime) * 0.05);
}

vec2 sincos(float x) { return vec2(sin(x), cos(x)); }

// Rotate a vector from one direction to another (Rodrigues' rotation)
vec3 rotate(vec3 vector, vec3 from, vec3 to) {
	float cosTheta = dot(from, to);
	if (abs(cosTheta) >= 0.9999) return (cosTheta < 0.0) ? -vector : vector;
	vec3 axis = normalize(cross(from, to));
	vec2 sc = vec2(sqrt(1.0 - cosTheta * cosTheta), cosTheta);
	return sc.y * vector +
		sc.x * cross(axis, vector) +
		(1.0 - sc.y) * dot(axis, vector) * axis;
}

// Single wave contribution: returns (height, derivative) for normal calculation
vec2 wavedx(vec2 position, vec2 direction, float frequency, float timeshift) {
	float x = dot(direction, position) * frequency + timeshift;
	x = mod(x, 2.0 * PI);
	float wave = exp(sin(x) - 1.0) * 0.5;
	float dx = wave * cos(x);
	return vec2(wave, -dx);
}

// Sum of 8 octaves of procedural waves.
// Returns vec3(height, gradient.x, gradient.y) for normal reconstruction.
vec3 waveHeightAndGradient(vec2 position) {
	float wavePhaseShift = length(position) * 0.1;
	float iter = 0.0;
	float frequency = 1.0;
	float timeMultiplier = 2.0;
	float weight = 1.0;
	float sumOfValues = 0.0;
	vec2 sumOfGradients = vec2(0.0);
	float sumOfWeights = 0.0;
	float wt = waveTime();
	// 8 iterations (weight decay makes later ones negligible)
	for (int i = 0; i < 8; i++) {
		float iterMod = mod(iter, 2.0 * PI);
		vec2 sc = sincos(iterMod); // sin in .x, cos in .y
		vec2 p = vec2(sc.x, sc.y);
		vec2 res = wavedx(position, p, frequency, wt * timeMultiplier + wavePhaseShift);
		position += p * res.y * weight * DRAG_MULT;
		sumOfValues += res.x * weight;
		sumOfGradients += p * res.y * weight * frequency;
		sumOfWeights += weight;
		weight *= 0.8;
		frequency *= 1.18;
		timeMultiplier *= 1.07;
		iter += 1232.399963;
	}
	float h = sumOfValues / sumOfWeights;
	vec2 g = sumOfGradients / sumOfWeights;
	return vec3(h, g.x, g.y);
}

// Height-only version (used when gradient not needed)
float waveHeight(vec2 position) {
	float wavePhaseShift = length(position) * 0.1;
	float iter = 0.0;
	float frequency = 1.0;
	float timeMultiplier = 2.0;
	float weight = 1.0;
	float sumOfValues = 0.0;
	float sumOfWeights = 0.0;
	float wt = waveTime();
	for (int i = 0; i < 8; i++) {
		float iterMod = mod(iter, 2.0 * PI);
		vec2 sc = sincos(iterMod);
		vec2 p = vec2(sc.x, sc.y);
		vec2 res = wavedx(position, p, frequency, wt * timeMultiplier + wavePhaseShift);
		position += p * res.y * weight * DRAG_MULT;
		sumOfValues += res.x * weight;
		sumOfWeights += weight;
		weight *= 0.8;
		frequency *= 1.18;
		timeMultiplier *= 1.07;
		iter += 1232.399963;
	}
	return sumOfValues / sumOfWeights;
}

// Compute wave surface normal from analytical gradient
vec3 waveNormal(vec2 pos, vec3 worldFaceNormal, float heightmapFactor) {
	vec3 hg = waveHeightAndGradient(pos.xy);
	float H = hg.x * WAVE_DEPTH * heightmapFactor;
	vec2 grad = hg.yz * WAVE_DEPTH * heightmapFactor;
	
	vec3 wn = normalize(vec3(-grad.x, 1.0, -grad.y));
	wn = rotate(wn, vec3(0.0, 1.0, 0.0), worldFaceNormal);
	return wn;
}

// Parallax-corrected wave normal for more accurate reflections at grazing angles
vec3 getWaterParallaxNormal(vec3 playerPos, vec3 worldNormal, float jitter, float heightmapFactor) {
	float ay = max(abs(playerPos.y), 0.0001);
	float fractionalDistance = (ay - WAVE_DEPTH) / ay;
	fractionalDistance = clamp(fractionalDistance, 0.0, 1.0);
	vec3 origin = playerPos * fractionalDistance;
	vec3 increment = (playerPos - origin) / float(WATER_PARALLAX_SAMPLES);

	vec3 rayPos = origin + increment * jitter;

	for (int i = 0; i < WATER_PARALLAX_SAMPLES; i++) {
		float wh = waveHeight(rayPos.xz + cameraPosition.xz) + playerPos.y;
		bool hit = ((playerPos.y < 0.0) == (rayPos.y < wh));
		if (hit) {
			increment *= 0.5;
		}
		rayPos += increment * (hit ? -1.0 : 1.0);
	}

	return waveNormal(rayPos.xz + cameraPosition.xz, worldNormal, 1.0);
}

// Public interface: get world-space wave normal with parallax correction
vec3 waterWaveNormalWorld(vec3 playerPos, vec3 viewDir, float jitter) {
	vec3 worldNormal = vec3(0.0, 1.0, 0.0);
	vec3 baseNView = normalize(mat3(gbufferModelView) * worldNormal);
	float heightmapFactor = sqrt(sin(PI * 0.5 * clamp01(abs(dot(baseNView, viewDir)))));
	return getWaterParallaxNormal(playerPos, worldNormal, jitter, heightmapFactor);
}


// ============================================================================
//  SKY RENDERING
// ============================================================================

// Nether sky: just use enhanced fog color
vec3 sampleNetherSky(vec3 worldDir, vec3 biomeFogColor) {
	vec3 linearFog = pow(biomeFogColor, vec3(2.2));
	return linearFog * 1.5;
}

// Apply rain desaturation to sky color
vec3 applyRainSkyDesat(vec3 c) {
	float r = pow(clamp(rainStrength, 0.0, 1.0), RAIN_SKY_CURVE);
	float l = dot(c, vec3(0.299, 0.587, 0.114));
	return mix(c, vec3(l), r * RAIN_SKY_DESAT_MAX);
}

// Get brightness multiplier for rainy weather
float rainSkyDimMult() {
	float r = pow(clamp(rainStrength, 0.0, 1.0), RAIN_SKY_CURVE);
	return mix(1.0, RAIN_SKY_DIM_MULT, r);
}

// Procedural sky with time-of-day colors, sun/moon glow, and stars.
// worldDir: normalized direction vector from camera
// Returns HDR sky color.
vec3 sampleProceduralSkybox(vec3 worldDir, vec3 fogColor, vec3 zenithTarget, float tSunrise, float tNoon, float tSunset, float tMidnight) {
	// Vertical gradient: fog color at horizon, zenith color above
	float yUp = clamp(worldDir.y, 0.0, 1.0);
	float zenithMix = pow(yUp, 0.35);
	vec3 targetSky = mix(fogColor, zenithTarget, zenithMix);
	
	// Add haze concentration at horizon
	float horizonSharpness = 6.0;
	float skyHaze = exp(-worldDir.y * worldDir.y * horizonSharpness);
	vec3 skyColor = mix(targetSky, fogColor, skyHaze);

	// rain dims and desaturates sky
	skyColor = applyRainSkyDesat(skyColor);
	skyColor *= rainSkyDimMult();

	// Sun and moon world-space directions
	vec3 sunDir = normalize((gbufferModelViewInverse * vec4(sunPosition, 0.0)).xyz);
	vec3 moonDir = normalize((gbufferModelViewInverse * vec4(moonPosition, 0.0)).xyz);

	float sunVisibility = 1.0 - tMidnight;
	float moonVisibility = tMidnight;

	// Sun glow (warm during sunrise/sunset)
	float sunDot = max(dot(worldDir, sunDir), 0.0);
	float sunDot2 = sunDot * sunDot;
	float sunGlow = sunDot2 * sunDot2 * 0.35;
	vec3 sunGlowColor = vec3(1.00, 0.92, 0.5);
	sunGlowColor = mix(sunGlowColor, vec3(1.0, 0.6, 0.3), tSunrise + tSunset);

	// Moon glow (cool blue-white)
	float moonDot = max(dot(worldDir, moonDir), 0.0);
	float moonDot2 = moonDot * moonDot;
	float moonDot4 = moonDot2 * moonDot2;
	float moonGlow = moonDot4 * moonDot4 * 0.18;
	vec3 moonGlowColor = vec3(0.7, 0.75, 0.9);

	// Procedural stars with twinkling
	vec3 starDir = floor(worldDir * 120.0);  // quantize direction for star grid
	float starHash = fract(sin(dot(starDir, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
	float starBright = step(0.992, starHash);
	float twinkleHash = fract(sin(dot(starDir, vec3(39.346, 11.135, 83.155))) * 65432.1234);
	float twinkle = sin(float(worldTime) * 0.01 + twinkleHash * 6.28) * 0.3 + 0.7;
	starBright *= twinkle;
	float nightTime = float(worldTime);
	float starNightFade = smoothstep(12000.0, 13500.0, nightTime) * (1.0 - smoothstep(22500.0, 24000.0, nightTime));
	float starHorizonFade = smoothstep(0.0, 0.3, worldDir.y);
	float starVisibility = starNightFade * starHorizonFade;
	vec3 starColor = vec3(0.9, 0.95, 1.0) * starBright * starVisibility * 0.4;

	vec3 celestials = vec3(0.0);
	celestials += sunGlowColor * sunGlow * sunVisibility;
	celestials += moonGlowColor * moonGlow * moonVisibility;
	celestials += starColor;

	float horizonFade = smoothstep(-0.05, 0.15, worldDir.y);
	celestials *= horizonFade;
	celestials *= (1.0 - rainStrength * 0.9);

	return skyColor + celestials;
}


// ============================================================================
//  SCREEN-SPACE REFLECTIONS (SSR)
// ============================================================================

// Raymarch through screen space to find reflection hit point.
// Returns true if hit found, with UV coordinates in hitUV.
bool ssrTrace(vec3 viewPos, vec3 viewDir, out vec2 hitUV) {
	if (viewDir.z > 0.0 && viewDir.z >= -viewPos.z) {
		return false;
	}

	// view space to screen space
	vec3 rayPos;
	{
		vec4 clip = gbufferProjection * vec4(viewPos, 1.0);
		clip.xyz /= clip.w;
		rayPos = vec3(clip.xy * 0.5 + 0.5, clip.z * 0.5 + 0.5);
	}
	vec3 rayDir;
	{
		vec4 clip = gbufferProjection * vec4(viewPos + viewDir, 1.0);
		clip.xyz /= clip.w;
		rayDir = vec3(clip.xy * 0.5 + 0.5, clip.z * 0.5 + 0.5);
	}
	float startZ = rayPos.z;

	rayDir -= rayPos;
	rayDir = normalize(rayDir);

	vec3 r = abs(sign(rayDir) - rayPos) / max(abs(rayDir), vec3(0.00001));
	float rayLength = min(r.x, min(r.y, r.z));
	
	int maxSteps = 16;
	float stepLength = rayLength / float(maxSteps * 3);
	vec3 rayStep = rayDir * stepLength;
	
	rayPos += rayStep * 0.85 + length(vec2(1.0 / viewWidth, 1.0 / viewHeight)) * rayDir;

	float vz = max(abs(viewPos.z), 0.0001);
	float depthLenience = max(abs(rayStep.z) * 3.0, 0.02 / (vz * vz));

	bool intersect = false;
	
	// coarse march
	for (int i = 0; i < 48; i++, rayPos += rayStep) {
		if (rayPos.x < 0.0 || rayPos.x > 1.0 || rayPos.y < 0.0 || rayPos.y > 1.0) return false;
		
		vec4 auxSample = texture2D(gaux1, rayPos.xy);
		if (auxSample.r > 0.5) continue;
		
		float depth = texture2D(gdepth, rayPos.xy).x;
		if (abs(depth - startZ) < 1e-5) continue;

		intersect =
			(depth < rayPos.z) &&
			(abs(depthLenience - (rayPos.z - depth)) < depthLenience) &&
			(depth < 1.0);

		if (intersect) break;
	}
	if (!intersect) return false;

	// binary refinement
	vec3 lastGoodPos = rayPos;
	vec3 stepDir = rayStep;
	for (int j = 0; j < 5; j++) {
		float depth = texture2D(gdepth, rayPos.xy).x;
		if (texture2D(gaux1, rayPos.xy).r > 0.5) depth = 1.0;
		float s = sign(depth - rayPos.z);
		if (s == 1.0 && depth < 1.0) lastGoodPos = rayPos;
		rayPos += s * stepDir;
		stepDir *= 0.5;
	}
	rayPos = lastGoodPos;

	hitUV = rayPos.xy;
	return true;
}




void main() {
	// cache primary per-pixel samples
	vec4 depthSample = texture2D(gdepth, texcoord.st);
	float rawDepth = depthSample.x;
	vec4 colorSample = texture2D(gcolor, texcoord.st);
	vec4 aux = texture2D(gaux1, texcoord.st);
	
	// Legacy pipeline note:
	// rawLightmap.* is often broken, so gbuffers write a lighting proxy into gnormal.r.
	// If it's missing/zero (e.g., passes that don't write gnormal), fall back to a
	// rough proxy from the already-lit scene color so PBR isn't hard-disabled.
	float skyLight = texture2D(gnormal, texcoord.st).r;
	if (skyLight < 0.0001) {
		float sceneLum = dot(colorSample.rgb, vec3(0.2126, 0.7152, 0.0722));
		skyLight = clamp(sceneLum * 1.25, 0.0, 1.0);
	}
	float skyLightFactor = smoothstep(0.18, 0.55, skyLight);
	skyLightFactor *= skyLightFactor;
	
	// dimension detection (nether has red-dominant fog)
	vec3 vanillaFogColor = gl_Fog.color.rgb;
	float fogRedness = vanillaFogColor.r / (vanillaFogColor.g + vanillaFogColor.b + 0.001);
	bool isNether = fogRedness > 1.5 && vanillaFogColor.r > 0.1;

	vec4 fragposition = gbufferProjectionInverse * vec4(texcoord.s * 2.0 - 1.0, texcoord.t * 2.0 - 1.0, 2.0 * rawDepth - 1.0, 1.0);
	fragposition /= fragposition.w;
	
	float drawdistance = SHADOW_RANGE;
	float drawdistancesquared = drawdistance * drawdistance;
	
	// sky vs land mask
	float landMask = aux.b;
	// water mask from gbuffers_water (aux.g=1, aux.a=1, aux.r=0)
	float waterMask = step(0.5, aux.g) * step(0.5, aux.a) * (1.0 - step(0.5, aux.r));
	float land = landMask;
	if (rawDepth >= 0.9999) {
		float a = colorSample.a;
		land = (a < 0.999) ? 1.0 : 0.0;
	}
	float distance = length(fragposition.xyz);

	// ========================================================================
	//  SHADOW MAPPING - PCF Soft Shadows
	// ========================================================================
	// Projects fragment to shadow space and samples shadow map with PCF filtering.
	// Uses distance-based edge fade to prevent hard shadow cutoff at render distance.
	
	float shadowValue = 1.0;
	
	// Only compute shadows in overworld and within render distance
	bool inShadowRange = !isNether && distance < drawdistance && distance > 0.1;
	if (inShadowRange) {
		// Transform fragment from view space to world space
		vec4 worldPos = gbufferModelViewInverse * fragposition;
		
		// Apply shadow offset to correct world-space misalignment
		worldPos.xyz += SHADOW_OFFSET;
		
		// Distance-based fade: squared distances for efficiency
		float horizDistSq = dot(worldPos.xz, worldPos.xz);
		float vertDistSq  = worldPos.y * worldPos.y;
		float maxDistSq = drawdistancesquared;
		
		// Only process if within vertical shadow bounds
		if (vertDistSq < maxDistSq) {
			// Transform to shadow view space, then to shadow clip space
			vec4 shadowViewPos = shadowModelView * worldPos;
			float fragmentShadowZ = -shadowViewPos.z;  // Depth from light's perspective
			
			vec4 shadowClipPos = shadowProjection * shadowViewPos;
			shadowClipPos /= shadowClipPos.w;
			
			// Convert from [-1,1] NDC to [0,1] texture coordinates
			vec2 shadowUV = shadowClipPos.xy * 0.5 + 0.5;
			
			// Bounds check: only sample if within shadow map
			bool inShadowBounds = fragmentShadowZ > 0.0 && 
			                      shadowUV.x > 0.0 && shadowUV.x < 1.0 && 
			                      shadowUV.y > 0.0 && shadowUV.y < 1.0;
			
			if (inShadowBounds) {
				// Edge fade factor: smooth falloff near shadow render distance
				float horizFade = 1.0 - horizDistSq / maxDistSq;
				float vertFade  = 1.0 - vertDistSq / maxDistSq;
				float edgeFade = pow(min(horizFade, vertFade), 0.35);  // Slight curve for smooth falloff
				
				#ifdef SHADOW_HQ
				
					// PCF filtering parameters
					vec2 jitterOffset = vec2(0.0, 0.00035 * SHADOW_SOFTNESS);
					float occlusionScale = 0.50 * SHADOW_INTENSITY;
					float depthThreshold = 0.9;
					float kernelRadius = 0.00005 * SHADOW_SOFTNESS;
					float biasedZ = fragmentShadowZ - SHADOW_BIAS;
					
					// Adaptive blur: sample coarse ring to detect penumbra region
					float penumbraDetect = 0.0;
					float coarseStep = 0.00085;
					
					// 8-sample penumbra detection ring
					#define PENUMBRA_TEST(dx, dy) edgeFade * max(clamp((biasedZ - texture2D(shadow, shadowUV + vec2(dx + jitterOffset.x, dy + jitterOffset.y)).z * SHADOW_DEPTH_SCALE) * occlusionScale * 0.01, 0.0, occlusionScale), 0.0)
					
					penumbraDetect = max(penumbraDetect, PENUMBRA_TEST( coarseStep,  coarseStep));
					penumbraDetect = max(penumbraDetect, PENUMBRA_TEST(-coarseStep,  coarseStep));
					penumbraDetect = max(penumbraDetect, PENUMBRA_TEST( coarseStep, -coarseStep));
					penumbraDetect = max(penumbraDetect, PENUMBRA_TEST(-coarseStep, -coarseStep));
					penumbraDetect = max(penumbraDetect, PENUMBRA_TEST( 0.0,         coarseStep));
					penumbraDetect = max(penumbraDetect, PENUMBRA_TEST( 0.0,        -coarseStep));
					penumbraDetect = max(penumbraDetect, PENUMBRA_TEST( coarseStep,  0.0));
					penumbraDetect = max(penumbraDetect, PENUMBRA_TEST(-coarseStep,  0.0));
					
					#undef PENUMBRA_TEST
					
					// Scale kernel by penumbra amount
					penumbraDetect = clamp(penumbraDetect - 0.006, 0.0, 1.0);
					kernelRadius *= penumbraDetect * 14.0;
					
					// PCF shadow accumulator with 48-sample disk pattern
					float occlusionFactor = occlusionScale / depthThreshold;
					
					#define PCF_TAP(dx, dy) (0.020833 - edgeFade * clamp((biasedZ - texture2D(shadow, shadowUV + vec2((dx)*kernelRadius+jitterOffset.x, (dy)*kernelRadius+jitterOffset.y)).z * SHADOW_DEPTH_SCALE) * occlusionFactor, 0.0, occlusionScale))
					
					// Ring 0 (center)
					shadowValue += PCF_TAP(0, 0);
					// Ring 1
					shadowValue += PCF_TAP(1, 0); shadowValue += PCF_TAP(0, 1);
					shadowValue += PCF_TAP(-1, 0); shadowValue += PCF_TAP(0, -1);
					shadowValue += PCF_TAP(1, 1); shadowValue += PCF_TAP(-1, 1);
					shadowValue += PCF_TAP(-1, -1); shadowValue += PCF_TAP(1, -1);
					// Ring 2
					shadowValue += PCF_TAP(2, 0); shadowValue += PCF_TAP(0, 2);
					shadowValue += PCF_TAP(-2, 0); shadowValue += PCF_TAP(0, -2);
					shadowValue += PCF_TAP(2, 1); shadowValue += PCF_TAP(1, 2);
					shadowValue += PCF_TAP(-1, 2); shadowValue += PCF_TAP(-2, 1);
					shadowValue += PCF_TAP(-2, -1); shadowValue += PCF_TAP(-1, -2);
					shadowValue += PCF_TAP(1, -2); shadowValue += PCF_TAP(2, -1);
					// Ring 3
					shadowValue += PCF_TAP(3, 0); shadowValue += PCF_TAP(0, 3);
					shadowValue += PCF_TAP(-3, 0); shadowValue += PCF_TAP(0, -3);
					shadowValue += PCF_TAP(2, 2); shadowValue += PCF_TAP(-2, 2);
					shadowValue += PCF_TAP(-2, -2); shadowValue += PCF_TAP(2, -2);
					shadowValue += PCF_TAP(3, 1); shadowValue += PCF_TAP(1, 3);
					shadowValue += PCF_TAP(-1, 3); shadowValue += PCF_TAP(-3, 1);
					shadowValue += PCF_TAP(-3, -1); shadowValue += PCF_TAP(-1, -3);
					shadowValue += PCF_TAP(1, -3); shadowValue += PCF_TAP(3, -1);
					// Ring 4
					shadowValue += PCF_TAP(4, 0); shadowValue += PCF_TAP(0, 4);
					shadowValue += PCF_TAP(-4, 0); shadowValue += PCF_TAP(0, -4);
					shadowValue += PCF_TAP(3, 2); shadowValue += PCF_TAP(2, 3);
					shadowValue += PCF_TAP(-2, 3); shadowValue += PCF_TAP(-3, 2);
					shadowValue += PCF_TAP(-3, -2); shadowValue += PCF_TAP(-2, -3);
					shadowValue += PCF_TAP(2, -3); shadowValue += PCF_TAP(3, -2);
					
					#undef PCF_TAP
					
					// Normalize and apply minimum brightness
					shadowValue = shadowValue / 48.0 + 0.97;
					shadowValue = clamp(shadowValue, 1.0 - 0.5 * SHADOW_INTENSITY, 1.0);
					
					// Sky pixels are fully lit (no shadows)
					if (land < 0.5) {
						shadowValue = 1.0;
					}
				
				#endif
			}
		}
	}
	
	// Combined shading factor (shadow + face shading)
	float shading = shadowValue;
	
	// pixel offsets for neighbor sampling
	float pw = 1.0 / viewWidth;
	float ph = 1.0 / viewHeight;
	
	// cached normal and neighbor positions
	float aHere = colorSample.a;
	vec3 cachedNormal = vec3(0.0, 0.0, 1.0);
	vec3 cachedPosR = vec3(0.0);
	vec3 cachedPosU = vec3(0.0);
	bool normalComputed = false;
	
	// time of day for light direction
	float wtimeEarly = float(worldTime);
	float isNight = step(12000.0, wtimeEarly) * (1.0 - step(23000.0, wtimeEarly));
	float useMoon = smoothstep(12000.0, 13000.0, wtimeEarly) * (1.0 - smoothstep(22500.0, 23500.0, wtimeEarly));
	
	// fallback face shading from screen-space normals
	if (land > 0.5 && rawDepth < 0.9999 && aHere >= 0.999) {
		vec3 posC = fragposition.xyz;
		cachedPosR = getViewPos(texcoord.st + vec2(pw, 0.0));
		cachedPosU = getViewPos(texcoord.st + vec2(0.0, ph));

		vec3 dx = cachedPosR - posC;
		vec3 dy = cachedPosU - posC;
		cachedNormal = normalize(cross(dx, dy));
		if (dot(cachedNormal, posC) > 0.0) cachedNormal = -cachedNormal;
		normalComputed = true;
		
		vec3 sunDir = normalize(sunPosition);
		vec3 moonDir = normalize(moonPosition);
		vec3 l = mix(sunDir, moonDir, useMoon);
		
		float ndotl = clamp(dot(cachedNormal, l), 0.0, 1.0);
		float faceShade = mix(0.55, 1.0, ndotl);
		shading = min(shading, faceShade);
	}

 
	vec4 color = colorSample;
	
	
	
// Compute sunlight exposure factor: maps shadow value [0-1] to sun illumination
// Uses smoothstep for soft transition between shadow and lit regions
float sunExposure = smoothstep(0.35, 0.95, shading);

// Rain reduces sun exposure to simulate overcast sky
if (rainStrength > 0.1) {
	sunExposure = 0.4;
}


  color.rgb = mix(color.rgb, (color.rgb*1.0), sunExposure * land);

	// time-of-day factors
	vec4 tod = getTimeOfDay(float(worldTime));
	float tSunrise  = tod.x;
	float tNoon     = tod.y;
	float tSunset   = tod.z;
	float tMidnight = tod.w;


	// ========================================================================
	//  TIME-OF-DAY LIGHTING TINTS
	// ========================================================================
	// Define color palettes for each time period, then blend based on world time.
	// Uses vec3 for cleaner code and unique color values.
	
	// Direct sunlight color at each time of day (warm sunrise/sunset, neutral noon, blue moonlight)
	vec3 directSunrise  = vec3(0.95, 0.82, 0.62);   // warm orange
	vec3 directNoon     = vec3(1.00, 0.98, 0.94);   // bright white-yellow
	vec3 directSunset   = vec3(0.92, 0.75, 0.58);   // warm orange-pink
	vec3 directMidnight = vec3(0.68, 0.72, 0.85);   // cool blue moonlight
	
	// Ambient/fill light color at each time of day
	// Keep shadows warm/neutral to avoid green/muddy interiors
	vec3 ambientSunrise  = vec3(0.92, 0.88, 0.82);  // warm neutral
	vec3 ambientNoon     = vec3(0.95, 0.93, 0.88);  // warm neutral (not blue!)
	vec3 ambientSunset   = vec3(0.90, 0.85, 0.78);  // warm muted
	vec3 ambientMidnight = vec3(0.80, 0.82, 0.88);  // slight cool, mostly neutral
	
	// Blend direct light by time weights
	vec3 directLight = directSunrise * tSunrise + directNoon * tNoon + 
	                   directSunset * tSunset + directMidnight * tMidnight;
	
	// Blend ambient light by time weights
	vec3 ambientLight = ambientSunrise * tSunrise + ambientNoon * tNoon + 
	                    ambientSunset * tSunset + ambientMidnight * tMidnight;
	
	// Slight warm bias to prevent green/muddy shadows
	// Green channel tends to dominate in dark areas due to eye sensitivity
	ambientLight.g *= 0.97;
	ambientLight.r *= 1.02;
	
	// Luminance normalization: ensure direct and ambient have similar perceived brightness
	float directLuma = dot(directLight, vec3(0.299, 0.587, 0.114));
	float ambientLuma = dot(ambientLight, vec3(0.299, 0.587, 0.114));
	float lumaNorm = (directLuma > 0.001) ? (ambientLuma / directLuma) : 1.0;
	directLight *= lumaNorm;

	// Apply time-of-day tinting: blend between ambient (shadow) and direct (lit) colors
	color.rgb = mix(color.rgb * ambientLight, color.rgb * directLight, sunExposure);

	// ========================================================================
	//  SKY COLOR ADJUSTMENT (non-terrain pixels)
	// ========================================================================
	// Adjust sky brightness based on time of day with unique blend factors
	if (land < 0.5) {
		// Sky brightness multipliers per time period (unique values, different approach)
		vec3 skyBrightMult = vec3(1.25, 1.30, 1.35) * tSunrise +   // warm dawn boost
		                     vec3(1.15, 1.10, 1.05) * tNoon +      // bright midday
		                     vec3(1.30, 1.25, 1.20) * tSunset +    // warm dusk boost
		                     vec3(0.95, 0.92, 0.88) * tMidnight;   // muted night
		vec3 skyOffset = vec3(-0.15) * tSunrise + vec3(-0.25) * tNoon + 
		                 vec3(-0.15) * tSunset + vec3(0.0) * tMidnight;
		color.rgb = color.rgb * skyBrightMult + skyOffset;
		
		// Rainy sky: desaturate and add gray cast
		if (rainStrength > 0.1) {
			float rainFactor = smoothstep(0.1, 1.0, rainStrength);
			vec3 grayBase = vec3(0.35, 0.38, 0.42);  // overcast gray
			float luma = dot(color.rgb, vec3(0.299, 0.587, 0.114));
			color.rgb = mix(color.rgb, vec3(luma) * 0.75 + grayBase * 0.4, rainFactor * 0.7);
		}
	}

	// Global rain dimming for all pixels
	if (rainStrength > 0.1) {
		color.rgb *= mix(1.0, 0.55, smoothstep(0.1, 1.0, rainStrength));
	}
	


	float noblur = aux.r;

	// fog blend
	float depthLinear = linearizeDepth(texcoord.st);
	
	// time-based sky/fog palette (matches final.fsh)

	// Palette targets (cartoony / vibrant) - must match final.fsh
	vec3 fogSunrise   = vec3(1.00, 0.55, 0.25);
	vec3 fogNoon      = vec3(0.75, 0.90, 1.00);
	vec3 fogSunset    = vec3(0.95, 0.45, 0.75);
	vec3 fogMidnight  = vec3(0.00, 0.00, 0.001);

	vec3 zenithSunrise  = vec3(0.38, 0.55, 1.00);
	vec3 zenithNoon     = vec3(0.14, 0.20, 0.98);
	vec3 zenithSunset   = vec3(0.22, 0.10, 0.58);
	vec3 zenithMidnight = vec3(0.0, 0.00, 0.0);

	vec3 fogTarget = fogSunrise * tSunrise + fogNoon * tNoon + fogSunset * tSunset + fogMidnight * tMidnight;
	vec3 zenithTarget = zenithSunrise * tSunrise + zenithNoon * tNoon + zenithSunset * tSunset + zenithMidnight * tMidnight;

	// blend vanilla fog with palette
	vec3 baseFog = gl_Fog.color.rgb;
	vec3 fogColor;
	if (isNether) {
		// nether fog
		fogColor = baseFog * vec3(1.2, 0.9, 0.8);
	} else {
		// overworld fog
		fogColor = mix(baseFog, fogTarget, 0.85);
	}
	
	// rain dims and desaturates fog
	if (!isNether) {
		fogColor = applyRainSkyDesat(fogColor);
		fogColor *= rainSkyDimMult();
		float rainT = pow(clamp(rainStrength, 0.0, 1.0), RAIN_SKY_CURVE);
		float extraDim = mix(1.0, 0.12, rainT);
		fogColor *= extraDim;
	}
	
	float fogFactor = 0.0;
	if (fogMode == GL_EXP) {
		fogFactor = 1.0 - clamp(exp(-gl_Fog.density * depthLinear), 0.0, 1.0);
	} else if (fogMode == GL_LINEAR) {
		fogFactor = clamp((depthLinear - gl_Fog.start) * gl_Fog.scale, 0.0, 1.0);
	} else {
		float fogStart = far * 0.6;
		float fogEnd = far;
		fogFactor = clamp((depthLinear - fogStart) / max(fogEnd - fogStart, 0.0001), 0.0, 1.0);
	}

	vec3 viewDir = normalize(fragposition.xyz);
	vec3 worldDir = normalize((gbufferModelViewInverse * vec4(viewDir, 0.0)).xyz);

	// sky view direction from screen ray
	vec4 screenRay = gbufferProjectionInverse * vec4(texcoord.s * 2.0 - 1.0, texcoord.t * 2.0 - 1.0, 1.0, 1.0);
	screenRay /= screenRay.w;
	vec3 skyViewDir = normalize(screenRay.xyz);
	vec3 skyWorldDir = normalize((gbufferModelViewInverse * vec4(skyViewDir, 0.0)).xyz);

	// apply shadow via soft light blend
	float shadingCurve = shading * shading * shading;
	float shadeBlend = clamp(shadingCurve * 0.5, 0.0, 0.5);
	vec3 shadedColor = softLight(color.rgb, shadeBlend);
	
	// extra shadow darkening
	float shadowDarken = mix(0.75, 1.0, shading);
	shadedColor *= shadowDarken;
	
	// cave/dark protection
	float baseLuma = dot(color.rgb, vec3(0.299, 0.587, 0.114));
	float veryDark = 1.0 - smoothstep(0.01, 0.06, baseLuma);
	float protect = clamp(veryDark * (1.0 - sunExposure), 0.0, 1.0);
	vec3 sceneColor = mix(shadedColor, color.rgb, protect);
	
	// ========================================================================
	//  EMISSIVE DETECTION & HDR BOOST
	// ========================================================================
	// Detect emissive light sources by comparing local lighting against a
	// blurred average. Pixels significantly brighter than their surroundings
	// receive an HDR boost for improved contrast in dark scenes.
	#ifdef EMISSIVE_DETECTION
	if (land > 0.5 && rawDepth < 0.9999) {
		// Get blur radius in pixels
		float emissiveBlurPx = EMISSIVE_BLUR_RADIUS * viewHeight;
		
		// Sample blurred lighting buffer
		float blurredLight = blurLightingBuffer(texcoord.st, emissiveBlurPx);
		
		// Compute emissive strength based on local vs average lighting
		float sceneLuma = dot(sceneColor, vec3(0.2126, 0.7152, 0.0722));
		float emissiveStrength = computeEmissiveStrength(texcoord.st, skyLight, blurredLight, sceneLuma);
		
		// Suppress emissive detection for sunlit pixels during daytime
		// If the sun is out (not midnight) and the pixel is receiving sunlight,
		// the brightness is from the sun, not from being an emissive source.
		float isDaytime = 1.0 - tMidnight;  // 1.0 during day, 0.0 at midnight
		float sunlitSuppression = sunExposure * isDaytime;  // high when sunlit during day
		emissiveStrength *= (1.0 - sunlitSuppression * 0.9);  // reduce emissive strength
		
		// Suppress dark pixels to avoid emissive bleed into unlit surfaces
		// Only actual light sources (bright pixels) should receive the boost
		float darkPixelSuppression = smoothstep(0.08, 0.25, sceneLuma);
		emissiveStrength *= darkPixelSuppression;
		
		#ifdef EMISSIVE_DEBUG
		// Debug: show emissive strength as white overlay
		// White = detected emissive, black = not emissive
		sceneColor = mix(sceneColor, vec3(1.0), emissiveStrength);
		#else
		// Apply emissive boost to scene color
		sceneColor = applyEmissiveBoost(sceneColor, emissiveStrength, skyLight);
		#endif
	}
	#endif
	
	vec3 outColor = sceneColor;
	if (land < 0.5) {
		// sky rendering
		if (isNether) {
			outColor = sampleNetherSky(skyWorldDir, vanillaFogColor);
		} else {
			// overworld sky: gradient + celestials
			float yUp = clamp(skyWorldDir.y, 0.0, 1.0);
			float zenithMix = pow(yUp, 0.35);
			vec3 targetSky = mix(fogColor, zenithTarget, zenithMix);

		// horizon haze
		float horizonSharpness = 6.0;
		float skyHaze = exp(-skyWorldDir.y * skyWorldDir.y * horizonSharpness);
		vec3 skyColor = mix(targetSky, fogColor, skyHaze);

		// boost sky saturation
		float skySatBoost = 1.45;
		float skyLumaPreSat = dot(skyColor, vec3(0.299, 0.587, 0.114));
		skyColor = mix(vec3(skyLumaPreSat), skyColor, skySatBoost);
		skyColor = max(skyColor, vec3(0.0));
		
				// rain desaturation
				skyColor = applyRainSkyDesat(skyColor);

		// sky intensity for HDR
		vec3 lumaW = vec3(0.299, 0.587, 0.114);
		float dayFactor = clamp(tSunrise + tNoon + tSunset, 0.0, 1.0);
		float curSkyLum = dot(skyColor, lumaW);
		float desiredSkyLum = 1.22 * tNoon + 0.85 * (tSunrise + tSunset) + 0.10 * tMidnight;
				desiredSkyLum *= rainSkyDimMult();
		float lumScale = desiredSkyLum / max(curSkyLum, 0.001);
		// rain allows darker sky
		float rainT = pow(clamp(rainStrength, 0.0, 1.0), RAIN_SKY_CURVE);
		float lumScaleMin = mix(0.5, 0.06, rainT);
		lumScale = clamp(lumScale, lumScaleMin, 4.0);
		skyColor *= lumScale;

		// procedural celestials
		
		vec3 sunDir = normalize((gbufferModelViewInverse * vec4(sunPosition, 0.0)).xyz);
		vec3 moonDir = normalize((gbufferModelViewInverse * vec4(moonPosition, 0.0)).xyz);
		
		// sun tangent frame for square projection
		vec3 sunRefUp = abs(sunDir.y) > 0.99 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
		vec3 sunRight = normalize(cross(sunDir, sunRefUp));
		vec3 sunUp = cross(sunRight, sunDir);
		vec3 toSun = skyWorldDir - sunDir * dot(skyWorldDir, sunDir);
		float sunU = dot(toSun, sunRight);
		float sunV = dot(toSun, sunUp);
		float sunSize = 0.075;
		
		// sun texture UV
		float sunTexU = (sunU / sunSize) * 0.5 + 0.5;
		float sunTexV = (sunV / sunSize) * 0.5 + 0.5;
		bool inSunQuad = sunTexU >= 0.0 && sunTexU <= 1.0 && sunTexV >= 0.0 && sunTexV <= 1.0 && dot(skyWorldDir, sunDir) > 0.0;
		
		// 8x8 sun texture
		vec3 sunTex[64];
		sunTex[0]=vec3(1.00,0.84,0.29); sunTex[1]=vec3(1.00,0.84,0.29); sunTex[2]=vec3(1.00,0.84,0.29); sunTex[3]=vec3(1.00,0.84,0.29);
		sunTex[4]=vec3(1.00,0.84,0.29); sunTex[5]=vec3(1.00,0.84,0.29); sunTex[6]=vec3(1.00,0.84,0.29); sunTex[7]=vec3(1.00,0.84,0.29);
		sunTex[8]=vec3(1.00,0.84,0.29); sunTex[9]=vec3(1.00,1.00,0.67); sunTex[10]=vec3(1.00,1.00,0.67); sunTex[11]=vec3(1.00,1.00,0.67);
		sunTex[12]=vec3(1.00,1.00,0.67); sunTex[13]=vec3(1.00,1.00,0.67); sunTex[14]=vec3(1.00,1.00,0.67); sunTex[15]=vec3(1.00,0.84,0.29);
		sunTex[16]=vec3(1.00,0.84,0.29); sunTex[17]=vec3(1.00,1.00,0.67); sunTex[18]=vec3(1.00,1.00,0.85); sunTex[19]=vec3(1.00,1.00,0.85);
		sunTex[20]=vec3(1.00,1.00,0.85); sunTex[21]=vec3(1.00,1.00,0.85); sunTex[22]=vec3(1.00,1.00,0.67); sunTex[23]=vec3(1.00,0.84,0.29);
		sunTex[24]=vec3(1.00,0.84,0.29); sunTex[25]=vec3(1.00,1.00,0.67); sunTex[26]=vec3(1.00,1.00,0.85); sunTex[27]=vec3(1.00,1.00,0.85);
		sunTex[28]=vec3(1.00,1.00,0.85); sunTex[29]=vec3(1.00,1.00,0.85); sunTex[30]=vec3(1.00,1.00,0.67); sunTex[31]=vec3(1.00,0.84,0.29);
		sunTex[32]=vec3(1.00,0.84,0.29); sunTex[33]=vec3(1.00,1.00,0.67); sunTex[34]=vec3(1.00,1.00,0.85); sunTex[35]=vec3(1.00,1.00,0.85);
		sunTex[36]=vec3(1.00,1.00,0.85); sunTex[37]=vec3(1.00,1.00,0.85); sunTex[38]=vec3(1.00,1.00,0.67); sunTex[39]=vec3(1.00,0.84,0.29);
		sunTex[40]=vec3(1.00,0.84,0.29); sunTex[41]=vec3(1.00,1.00,0.67); sunTex[42]=vec3(1.00,1.00,0.85); sunTex[43]=vec3(1.00,1.00,0.85);
		sunTex[44]=vec3(1.00,1.00,0.85); sunTex[45]=vec3(1.00,1.00,0.85); sunTex[46]=vec3(1.00,1.00,0.67); sunTex[47]=vec3(1.00,0.84,0.29);
		sunTex[48]=vec3(1.00,0.84,0.29); sunTex[49]=vec3(1.00,1.00,0.67); sunTex[50]=vec3(1.00,1.00,0.67); sunTex[51]=vec3(1.00,1.00,0.67);
		sunTex[52]=vec3(1.00,1.00,0.67); sunTex[53]=vec3(1.00,1.00,0.67); sunTex[54]=vec3(1.00,1.00,0.67); sunTex[55]=vec3(1.00,0.84,0.29);
		sunTex[56]=vec3(1.00,0.84,0.29); sunTex[57]=vec3(1.00,0.84,0.29); sunTex[58]=vec3(1.00,0.84,0.29); sunTex[59]=vec3(1.00,0.84,0.29);
		sunTex[60]=vec3(1.00,0.84,0.29); sunTex[61]=vec3(1.00,0.84,0.29); sunTex[62]=vec3(1.00,0.84,0.29); sunTex[63]=vec3(1.00,0.84,0.29);
		
		// sample sun texture (binary search)
		vec3 sunTexColor = vec3(0.0);
		float sunDisc = 0.0;
		if (inSunQuad) {
			int sunPx = int(clamp(sunTexU * 8.0, 0.0, 7.0));
			int sunPy = int(clamp((1.0 - sunTexV) * 8.0, 0.0, 7.0));
			int idx = sunPy * 8 + sunPx;
			if (idx < 32) {
				if (idx < 16) {
					if (idx < 8) {
						if (idx < 4) {
							if (idx < 2) { sunTexColor = (idx == 0) ? sunTex[0] : sunTex[1]; }
							else { sunTexColor = (idx == 2) ? sunTex[2] : sunTex[3]; }
						} else {
							if (idx < 6) { sunTexColor = (idx == 4) ? sunTex[4] : sunTex[5]; }
							else { sunTexColor = (idx == 6) ? sunTex[6] : sunTex[7]; }
						}
					} else {
						if (idx < 12) {
							if (idx < 10) { sunTexColor = (idx == 8) ? sunTex[8] : sunTex[9]; }
							else { sunTexColor = (idx == 10) ? sunTex[10] : sunTex[11]; }
						} else {
							if (idx < 14) { sunTexColor = (idx == 12) ? sunTex[12] : sunTex[13]; }
							else { sunTexColor = (idx == 14) ? sunTex[14] : sunTex[15]; }
						}
					}
				} else {
					if (idx < 24) {
						if (idx < 20) {
							if (idx < 18) { sunTexColor = (idx == 16) ? sunTex[16] : sunTex[17]; }
							else { sunTexColor = (idx == 18) ? sunTex[18] : sunTex[19]; }
						} else {
							if (idx < 22) { sunTexColor = (idx == 20) ? sunTex[20] : sunTex[21]; }
							else { sunTexColor = (idx == 22) ? sunTex[22] : sunTex[23]; }
						}
					} else {
						if (idx < 28) {
							if (idx < 26) { sunTexColor = (idx == 24) ? sunTex[24] : sunTex[25]; }
							else { sunTexColor = (idx == 26) ? sunTex[26] : sunTex[27]; }
						} else {
							if (idx < 30) { sunTexColor = (idx == 28) ? sunTex[28] : sunTex[29]; }
							else { sunTexColor = (idx == 30) ? sunTex[30] : sunTex[31]; }
						}
					}
				}
			} else {
				if (idx < 48) {
					if (idx < 40) {
						if (idx < 36) {
							if (idx < 34) { sunTexColor = (idx == 32) ? sunTex[32] : sunTex[33]; }
							else { sunTexColor = (idx == 34) ? sunTex[34] : sunTex[35]; }
						} else {
							if (idx < 38) { sunTexColor = (idx == 36) ? sunTex[36] : sunTex[37]; }
							else { sunTexColor = (idx == 38) ? sunTex[38] : sunTex[39]; }
						}
					} else {
						if (idx < 44) {
							if (idx < 42) { sunTexColor = (idx == 40) ? sunTex[40] : sunTex[41]; }
							else { sunTexColor = (idx == 42) ? sunTex[42] : sunTex[43]; }
						} else {
							if (idx < 46) { sunTexColor = (idx == 44) ? sunTex[44] : sunTex[45]; }
							else { sunTexColor = (idx == 46) ? sunTex[46] : sunTex[47]; }
						}
					}
				} else {
					if (idx < 56) {
						if (idx < 52) {
							if (idx < 50) { sunTexColor = (idx == 48) ? sunTex[48] : sunTex[49]; }
							else { sunTexColor = (idx == 50) ? sunTex[50] : sunTex[51]; }
						} else {
							if (idx < 54) { sunTexColor = (idx == 52) ? sunTex[52] : sunTex[53]; }
							else { sunTexColor = (idx == 54) ? sunTex[54] : sunTex[55]; }
						}
					} else {
						if (idx < 60) {
							if (idx < 58) { sunTexColor = (idx == 56) ? sunTex[56] : sunTex[57]; }
							else { sunTexColor = (idx == 58) ? sunTex[58] : sunTex[59]; }
						} else {
							if (idx < 62) { sunTexColor = (idx == 60) ? sunTex[60] : sunTex[61]; }
							else { sunTexColor = (idx == 62) ? sunTex[62] : sunTex[63]; }
						}
					}
				}
			}
			sunDisc = 1.0;
		}
		
		// sun glow
		float sunSquareDist = max(abs(sunU), abs(sunV));
		float sunCircleDist = sqrt(sunU * sunU + sunV * sunV);
		float sunDotSky = max(dot(skyWorldDir, sunDir), 0.0);
		float sunGlowWide = sunDotSky * sunDotSky * sunDotSky * sunDotSky * 0.3;
		float sunGlowCloseTmp = max(1.0 - sunCircleDist / (sunSize * 3.0), 0.0);
		float sunGlowClose = sunGlowCloseTmp * sunGlowCloseTmp * 0.6;
		sunGlowClose *= step(0.0, dot(skyWorldDir, sunDir));
		float sunGlow = sunGlowWide + sunGlowClose;
		vec3 sunGlowColor = vec3(1.00, 0.92, 0.5);
		sunGlowColor = mix(sunGlowColor, vec3(1.0, 0.6, 0.3), tSunrise + tSunset);
		
		// moon
		vec3 moonRefUp = abs(moonDir.y) > 0.99 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
		vec3 moonRight = normalize(cross(moonDir, moonRefUp));
		vec3 moonUp = cross(moonRight, moonDir);
		vec3 toMoon = skyWorldDir - moonDir * dot(skyWorldDir, moonDir);
		float moonU = dot(toMoon, moonRight);
		float moonV = dot(toMoon, moonUp);
		float moonSize = 0.05;
		
		float moonTexU = (moonU / moonSize) * 0.5 + 0.5;
		float moonTexV = (moonV / moonSize) * 0.5 + 0.5;
		bool inMoonQuad = moonTexU >= 0.0 && moonTexU <= 1.0 && moonTexV >= 0.0 && moonTexV <= 1.0 && dot(skyWorldDir, moonDir) > 0.0;
		
		// 8x8 moon texture
		vec3 moonTex[64];
		moonTex[0]=vec3(0.37,0.40,0.48); moonTex[1]=vec3(0.31,0.34,0.40); moonTex[2]=vec3(0.85,0.89,1.00); moonTex[3]=vec3(0.56,0.59,0.65);
		moonTex[4]=vec3(0.85,0.89,1.00); moonTex[5]=vec3(0.85,0.89,1.00); moonTex[6]=vec3(0.85,0.89,1.00); moonTex[7]=vec3(0.85,0.89,1.00);
		moonTex[8]=vec3(0.37,0.40,0.48); moonTex[9]=vec3(0.37,0.40,0.48); moonTex[10]=vec3(0.85,0.89,1.00); moonTex[11]=vec3(0.85,0.89,1.00);
		moonTex[12]=vec3(0.85,0.89,1.00); moonTex[13]=vec3(0.69,0.72,0.80); moonTex[14]=vec3(0.56,0.59,0.65); moonTex[15]=vec3(0.85,0.89,1.00);
		moonTex[16]=vec3(0.37,0.40,0.48); moonTex[17]=vec3(0.45,0.49,0.58); moonTex[18]=vec3(0.85,0.89,1.00); moonTex[19]=vec3(0.69,0.72,0.80);
		moonTex[20]=vec3(0.85,0.89,1.00); moonTex[21]=vec3(0.69,0.72,0.80); moonTex[22]=vec3(0.56,0.59,0.65); moonTex[23]=vec3(0.85,0.89,1.00);
		moonTex[24]=vec3(0.31,0.34,0.40); moonTex[25]=vec3(0.31,0.34,0.40); moonTex[26]=vec3(0.85,0.89,1.00); moonTex[27]=vec3(0.69,0.72,0.80);
		moonTex[28]=vec3(0.56,0.59,0.65); moonTex[29]=vec3(0.69,0.72,0.80); moonTex[30]=vec3(0.69,0.72,0.80); moonTex[31]=vec3(0.85,0.89,1.00);
		moonTex[32]=vec3(0.37,0.40,0.48); moonTex[33]=vec3(0.37,0.40,0.48); moonTex[34]=vec3(0.69,0.72,0.80); moonTex[35]=vec3(0.69,0.72,0.80);
		moonTex[36]=vec3(0.56,0.59,0.65); moonTex[37]=vec3(0.85,0.89,1.00); moonTex[38]=vec3(0.85,0.89,1.00); moonTex[39]=vec3(0.56,0.59,0.65);
		moonTex[40]=vec3(0.37,0.40,0.48); moonTex[41]=vec3(0.45,0.49,0.58); moonTex[42]=vec3(0.85,0.89,1.00); moonTex[43]=vec3(0.69,0.72,0.80);
		moonTex[44]=vec3(0.69,0.72,0.80); moonTex[45]=vec3(0.69,0.72,0.80); moonTex[46]=vec3(0.69,0.72,0.80); moonTex[47]=vec3(0.85,0.89,1.00);
		moonTex[48]=vec3(0.37,0.40,0.48); moonTex[49]=vec3(0.31,0.34,0.40); moonTex[50]=vec3(0.45,0.49,0.58); moonTex[51]=vec3(0.37,0.40,0.48);
		moonTex[52]=vec3(0.37,0.40,0.48); moonTex[53]=vec3(0.37,0.40,0.48); moonTex[54]=vec3(0.31,0.34,0.40); moonTex[55]=vec3(0.45,0.49,0.58);
		moonTex[56]=vec3(0.37,0.40,0.48); moonTex[57]=vec3(0.37,0.40,0.48); moonTex[58]=vec3(0.37,0.40,0.48); moonTex[59]=vec3(0.37,0.40,0.48);
		moonTex[60]=vec3(0.31,0.34,0.40); moonTex[61]=vec3(0.37,0.40,0.48); moonTex[62]=vec3(0.37,0.40,0.48); moonTex[63]=vec3(0.37,0.40,0.48);
		
		// sample moon texture (binary search)
		vec3 moonTexColor = vec3(0.0);
		float moonDisc = 0.0;
		if (inMoonQuad) {
			int moonPx = int(clamp(moonTexU * 8.0, 0.0, 7.0));
			int moonPy = int(clamp((1.0 - moonTexV) * 8.0, 0.0, 7.0));
			int idx = moonPy * 8 + moonPx;
			if (idx < 32) {
				if (idx < 16) {
					if (idx < 8) {
						if (idx < 4) {
							if (idx < 2) { moonTexColor = (idx == 0) ? moonTex[0] : moonTex[1]; }
							else { moonTexColor = (idx == 2) ? moonTex[2] : moonTex[3]; }
						} else {
							if (idx < 6) { moonTexColor = (idx == 4) ? moonTex[4] : moonTex[5]; }
							else { moonTexColor = (idx == 6) ? moonTex[6] : moonTex[7]; }
						}
					} else {
						if (idx < 12) {
							if (idx < 10) { moonTexColor = (idx == 8) ? moonTex[8] : moonTex[9]; }
							else { moonTexColor = (idx == 10) ? moonTex[10] : moonTex[11]; }
						} else {
							if (idx < 14) { moonTexColor = (idx == 12) ? moonTex[12] : moonTex[13]; }
							else { moonTexColor = (idx == 14) ? moonTex[14] : moonTex[15]; }
						}
					}
				} else {
					if (idx < 24) {
						if (idx < 20) {
							if (idx < 18) { moonTexColor = (idx == 16) ? moonTex[16] : moonTex[17]; }
							else { moonTexColor = (idx == 18) ? moonTex[18] : moonTex[19]; }
						} else {
							if (idx < 22) { moonTexColor = (idx == 20) ? moonTex[20] : moonTex[21]; }
							else { moonTexColor = (idx == 22) ? moonTex[22] : moonTex[23]; }
						}
					} else {
						if (idx < 28) {
							if (idx < 26) { moonTexColor = (idx == 24) ? moonTex[24] : moonTex[25]; }
							else { moonTexColor = (idx == 26) ? moonTex[26] : moonTex[27]; }
						} else {
							if (idx < 30) { moonTexColor = (idx == 28) ? moonTex[28] : moonTex[29]; }
							else { moonTexColor = (idx == 30) ? moonTex[30] : moonTex[31]; }
						}
					}
				}
			} else {
				if (idx < 48) {
					if (idx < 40) {
						if (idx < 36) {
							if (idx < 34) { moonTexColor = (idx == 32) ? moonTex[32] : moonTex[33]; }
							else { moonTexColor = (idx == 34) ? moonTex[34] : moonTex[35]; }
						} else {
							if (idx < 38) { moonTexColor = (idx == 36) ? moonTex[36] : moonTex[37]; }
							else { moonTexColor = (idx == 38) ? moonTex[38] : moonTex[39]; }
						}
					} else {
						if (idx < 44) {
							if (idx < 42) { moonTexColor = (idx == 40) ? moonTex[40] : moonTex[41]; }
							else { moonTexColor = (idx == 42) ? moonTex[42] : moonTex[43]; }
						} else {
							if (idx < 46) { moonTexColor = (idx == 44) ? moonTex[44] : moonTex[45]; }
							else { moonTexColor = (idx == 46) ? moonTex[46] : moonTex[47]; }
						}
					}
				} else {
					if (idx < 56) {
						if (idx < 52) {
							if (idx < 50) { moonTexColor = (idx == 48) ? moonTex[48] : moonTex[49]; }
							else { moonTexColor = (idx == 50) ? moonTex[50] : moonTex[51]; }
						} else {
							if (idx < 54) { moonTexColor = (idx == 52) ? moonTex[52] : moonTex[53]; }
							else { moonTexColor = (idx == 54) ? moonTex[54] : moonTex[55]; }
						}
					} else {
						if (idx < 60) {
							if (idx < 58) { moonTexColor = (idx == 56) ? moonTex[56] : moonTex[57]; }
							else { moonTexColor = (idx == 58) ? moonTex[58] : moonTex[59]; }
						} else {
							if (idx < 62) { moonTexColor = (idx == 60) ? moonTex[60] : moonTex[61]; }
							else { moonTexColor = (idx == 62) ? moonTex[62] : moonTex[63]; }
						}
					}
				}
			}
			moonDisc = 1.0;
		}
		
		// moon glow
		float moonSquareDist = max(abs(moonU), abs(moonV));
		float moonCircleDist = sqrt(moonU * moonU + moonV * moonV);
		float moonDotSky = max(dot(skyWorldDir, moonDir), 0.0);
		float moonDot2 = moonDotSky * moonDotSky;
		float moonDot4 = moonDot2 * moonDot2;
		float moonGlowWide = moonDot4 * moonDot4 * 0.15;
		float moonGlowCloseTmp = max(1.0 - moonCircleDist / (moonSize * 3.0), 0.0);
		float moonGlowClose = moonGlowCloseTmp * moonGlowCloseTmp * 0.4;
		moonGlowClose *= step(0.0, dot(skyWorldDir, moonDir));
		float moonGlow = moonGlowWide + moonGlowClose;
		vec3 moonGlowColor = vec3(0.7, 0.75, 0.9);
		
		// procedural stars
		vec3 starDir = floor(skyWorldDir * 120.0);
		float starHash = fract(sin(dot(starDir, vec3(12.9898, 78.233, 45.164))) * 43758.5453);
		float starBright = step(0.992, starHash);
		float twinkleHash = fract(sin(dot(starDir, vec3(39.346, 11.135, 83.155))) * 65432.1234);
		float twinkle = sin(float(worldTime) * 0.01 + twinkleHash * 6.28) * 0.3 + 0.7;
		starBright *= twinkle;
		// stars visible during night, fade at horizon
		float nightTime = float(worldTime);
		float starNightFade = smoothstep(12000.0, 13500.0, nightTime) * (1.0 - smoothstep(22500.0, 24000.0, nightTime));
		float starHorizonFade = smoothstep(0.0, 0.3, skyWorldDir.y);
		float starVisibility = starNightFade * starHorizonFade;
		vec3 starColor = vec3(0.9, 0.95, 1.0) * starBright * starVisibility * 0.4;
		
		// combine sky + celestials
		float sunVisibility = 1.0 - tMidnight;
		vec3 celestials = sunTexColor * sunDisc * sunVisibility;
		celestials += sunGlowColor * sunGlow * sunVisibility;
		celestials += moonTexColor * moonDisc * tMidnight;
		celestials += moonGlowColor * moonGlow * tMidnight;
		celestials += starColor;
		
		// fade celestials near horizon
		float horizonFade = smoothstep(-0.05, 0.15, skyWorldDir.y);
		celestials *= horizonFade;
		
		// fade in rain
		celestials *= (1.0 - rainStrength * 0.9);
		
		outColor = skyColor + celestials;
		} // end Overworld sky
	} else {
		// water reflections (SSR)
		if (waterMask > 0.5 && aHere < 0.999) {
			vec3 viewPos = fragposition.xyz;
			vec3 V = normalize(viewPos);
			vec4 worldPos4 = gbufferModelViewInverse * vec4(viewPos, 1.0);
			vec3 worldPos = worldPos4.xyz;
			float jitter = fract(sin(dot(texcoord.st * vec2(viewWidth, viewHeight), vec2(12.9898, 78.233)) + frameTimeCounter) * 43758.5453);

			// geometric normal for vertical surface detection
			vec3 geoNormal;
			{
				vec3 posC = viewPos;
				vec3 posR = getViewPos(texcoord.st + vec2(pw, 0.0));
				vec3 posU = getViewPos(texcoord.st + vec2(0.0, ph));
				vec3 dx = posR - posC;
				vec3 dy = posU - posC;
				geoNormal = normalize(cross(dx, dy));
				if (dot(geoNormal, posC) > 0.0) geoNormal = -geoNormal;
			}
			// check if surface is horizontal
			vec3 geoNWorld = normalize((gbufferModelViewInverse * vec4(geoNormal, 0.0)).xyz);
			float isHorizontal = abs(geoNWorld.y);

			vec3 waveNWorld = waterWaveNormalWorld(worldPos, V, jitter);
			vec3 waveNView = normalize((gbufferModelView * vec4(waveNWorld, 0.0)).xyz);
			vec3 R = normalize(reflect(V, waveNView));

			// fresnel
			float NoV = clamp(dot(waveNView, -V), 0.0, 1.0);
			float fresnelX = 1.0 - NoV;
			float fresnelX2 = fresnelX * fresnelX;
			float fresnel = fresnelX2 * fresnelX2 * fresnelX;

			// Skybox fallback in reflected direction (prevents dark spots on SSR miss)
			vec3 Rworld = normalize((gbufferModelViewInverse * vec4(R, 0.0)).xyz);
			vec3 skyRefl = isNether ? sampleNetherSky(Rworld, vanillaFogColor) : sampleProceduralSkybox(Rworld, fogColor, zenithTarget, tSunrise, tNoon, tSunset, tMidnight);
			vec3 refl = skyRefl;
			float hitFade = 0.0;
			vec2 hitUV = vec2(0.0);
			// trace SSR
			if (R.z < -0.02 && ssrTrace(viewPos, R, hitUV)) {
				float hitDepth = texture2D(gdepth, hitUV).x;
				vec4 hitAux = texture2D(gaux1, hitUV);
				float hitWater = step(0.5, hitAux.g) * step(0.5, hitAux.a);
				float hitHand = step(0.5, hitAux.r);
				// classify hit (same rules as main pass)
				float hitLand = hitAux.b;
				if (hitDepth >= 0.9999) {
					float a = texture2D(gcolor, hitUV).a;
					hitLand = (a < 0.999) ? 1.0 : 0.0;
				}
				// accept geometry hits, sky falls back to skybox
				if (hitLand >= 0.5 && hitHand < 0.5) {
					vec3 hitColor = texture2D(gcolor, hitUV).rgb;
					float hitDepthLinear = linearizeDepth(hitUV);
					float hitFog = 0.0;
					if (fogMode == GL_EXP) {
						hitFog = 1.0 - clamp(exp(-gl_Fog.density * hitDepthLinear), 0.0, 1.0);
					} else if (fogMode == GL_LINEAR) {
						hitFog = clamp((hitDepthLinear - gl_Fog.start) * gl_Fog.scale, 0.0, 1.0);
					} else {
						float fogStart = far * 0.6;
						float fogEnd = far;
						hitFog = clamp((hitDepthLinear - fogStart) / max(fogEnd - fogStart, 0.0001), 0.0, 1.0);
					}
					hitColor = mix(hitColor, fogColor, hitFog);
					// edge fade
					float edge = min(min(hitUV.x, 1.0 - hitUV.x), min(hitUV.y, 1.0 - hitUV.y));
					float edgeFade = smoothstep(0.02, 0.10, edge);
					hitFade = edgeFade;
					refl = mix(skyRefl, hitColor, hitFade);
				}
			}

			// reflection strength (fresnel + cave suppression)
			float reflAmt = (0.15 + 0.85 * fresnel) * (1.0 - rainStrength * 0.35);
			// use blue intensity as proxy for outdoors
			float blueIntensity = clamp(colorSample.b - 0.5 * (colorSample.r + colorSample.g), 0.0, 1.0);
			float blueFade = smoothstep(0.04, 0.22, blueIntensity);
			blueFade *= blueFade;
			reflAmt *= (0.10 + 0.90 * blueFade);
			// kill reflections on vertical surfaces
			float verticalFade = smoothstep(0.3, 0.7, isHorizontal);
			reflAmt *= verticalFade;
			reflAmt = min(reflAmt, 0.55);
			sceneColor = mix(sceneColor, refl, clamp(reflAmt, 0.0, 1.0));
		}

		// PBRlite specular (GGX microfacet)
		if (waterMask < 0.5 && land > 0.5 && rawDepth < 0.9999 && aux.r < 0.5) {
			vec3 baseAlbedo = clamp(colorSample.rgb, 0.0, 1.0);
			float smoothness = clamp(aux.a * PBR_SMOOTHNESS_MULT, 0.0, 1.0);
			float metallic = clamp(aux.g / 0.49, 0.0, 1.0);
			float dielectricF0 = clamp(aux.r / 0.49, 0.02, 1.0);
			// Shadow visibility from shadowmap (0=shadowed, 1=lit). This ensures shadowed
			// regions do not receive sun/env/moon specular highlights.
			float shadowVis = clamp(sunExposure, 0.0, 1.0);
			
			// anti-sparkle for noisy cutout geometry
			float depthVar = fwidth(rawDepth);
			float sparkleSuppress = smoothstep(0.0010, 0.0150, depthVar);
			smoothness *= (1.0 - sparkleSuppress * 0.90);
			
			// early-out for rough surfaces
			if (smoothness > 0.005) {
				vec3 viewPos = fragposition.xyz;
				vec3 nViewPos = normalize(viewPos);
				vec3 V = -nViewPos;
				
				vec2 uv = texcoord.st;
				vec3 posC = viewPos;
				float filterStrength = clamp(sparkleSuppress * 0.8, 0.0, 0.6);
				vec4 filterResult = edgeAwareNormal(uv, pw, ph, filterStrength, posC, cachedPosR, cachedPosU);
				vec3 N = filterResult.xyz;
				float filterConfidence = filterResult.w;
				
				float NoV = max(dot(N, V), 0.001);
				float roughness = sqr(1.0 - smoothness);
				roughness = mix(roughness, max(roughness, 0.6), (1.0 - filterConfidence) * filterStrength);
				roughness = clamp(roughness, 0.04, 1.0);
				
				vec3 F0 = mix(vec3(dielectricF0), baseAlbedo, metallic);
				F0 = mix(F0, vec3(0.04), sparkleSuppress * 0.6);
				vec3 reflectTint = mix(vec3(1.0), baseAlbedo, metallic);
				
				// environment reflection
				vec3 R = reflect(nViewPos, N);
				R = normalize(mix(R, N, clamp(roughness, 0.0, 1.0) * 0.5));
				vec3 Rworld = normalize((gbufferModelViewInverse * vec4(R, 0.0)).xyz);
				vec3 envColor = isNether ? sampleNetherSky(Rworld, vanillaFogColor) : sampleProceduralSkybox(Rworld, fogColor, zenithTarget, tSunrise, tNoon, tSunset, tMidnight);
				vec3 Fenv = fresnelSchlick(NoV, F0);
				float envStrength = (1.0 - roughness) * (1.0 - rainStrength * 0.5);
				envStrength *= skyLightFactor;
				envStrength *= shadowVis;
				envStrength *= PBR_ENV_STRENGTH;
				vec3 envContribution = envColor * reflectTint * Fenv * envStrength;
				sceneColor += envContribution;
				
				// sun specular (GGX area light)
				vec3 L = normalize(sunPosition);
				float NoL = max(dot(N, L), 0.0);
				if (NoL > 0.001 && sunExposure > 0.01) {
					vec3 sunSpec = GGX_Specular_AreaLight(N, V, L, smoothness, F0, SUN_ANGULAR_RADIUS);
					sunSpec = sunSpec / (vec3(1.0) + sunSpec * 0.5);
					vec3 sunSpecColor = mix(vec3(0.4, 0.45, 0.6), vec3(1.0, 0.95, 0.85), 1.0 - tMidnight);
					sunSpecColor = mix(sunSpecColor, vec3(1.0, 0.7, 0.4), tSunrise + tSunset);
					float sunStrength = sunExposure * (1.0 - rainStrength * 0.6);
					sunStrength *= skyLightFactor;
					sunStrength *= shadowVis;
					sunStrength *= PBR_SUN_STRENGTH;
					sceneColor += sunSpecColor * sunSpec * sunStrength;
				}
				
				// moon specular
				vec3 moonL = normalize(moonPosition);
				float moonNoL = max(dot(N, moonL), 0.0);
				float moonShadow = (tMidnight > 0.5) ? sunExposure : 0.0;
				if (moonNoL > 0.001 && tMidnight > 0.5 && moonShadow > 0.01) {
					vec3 moonSpec = GGX_Specular_AreaLight(N, V, moonL, smoothness, F0, MOON_ANGULAR_RADIUS);
					vec3 moonSpecColor = vec3(0.5, 0.55, 0.7);
					float moonStrength = 0.12 * tMidnight * moonShadow * (1.0 - rainStrength * 0.8);
					moonStrength *= skyLightFactor;
					moonStrength *= shadowVis;
					moonStrength *= PBR_MOON_STRENGTH;
					sceneColor += moonSpecColor * moonSpec * moonStrength;
				}
			}
		}

		// fog applied in final.fsh after ICGI
		outColor = sceneColor;
	}

	// cloud brightening (blend with sky behind)
	vec4 cloudWorldPos = gbufferModelViewInverse * fragposition;
	float worldY = cloudWorldPos.y + cameraPosition.y;
	bool isCloud = (aHere < 0.999) && (aHere > 0.01) && (worldY > 107.0);
	if (isCloud) {
		// compute sky color for this direction
		vec4 screenRayCloud = gbufferProjectionInverse * vec4(texcoord.s * 2.0 - 1.0, texcoord.t * 2.0 - 1.0, 1.0, 1.0);
		screenRayCloud /= screenRayCloud.w;
		vec3 cloudWorldDir = normalize((gbufferModelViewInverse * vec4(screenRayCloud.xyz, 0.0)).xyz);
		
		float yUpCloud = clamp(cloudWorldDir.y, 0.0, 1.0);
		float zenithMixCloud = pow(yUpCloud, 0.35);
		vec3 cloudSkyColor = mix(fogColor, zenithTarget, zenithMixCloud);
		
		// match main sky processing
		float cloudSkySatBoost = 1.45;
		float cloudSkyLumaPreSat = dot(cloudSkyColor, vec3(0.299, 0.587, 0.114));
		cloudSkyColor = mix(vec3(cloudSkyLumaPreSat), cloudSkyColor, cloudSkySatBoost);
		cloudSkyColor = max(cloudSkyColor, vec3(0.0));
		cloudSkyColor = applyRainSkyDesat(cloudSkyColor);
		
		// HDR scaling
		float cloudDayFactor = clamp(tSunrise + tNoon + tSunset, 0.0, 1.0);
		float cloudCurSkyLum = dot(cloudSkyColor, vec3(0.299, 0.587, 0.114));
		float cloudDesiredSkyLum = 1.22 * tNoon + 0.85 * (tSunrise + tSunset) + 0.10 * tMidnight;
		cloudDesiredSkyLum *= rainSkyDimMult();
		float cloudLumScale = cloudDesiredSkyLum / max(cloudCurSkyLum, 0.001);
		float cloudRainT = pow(clamp(rainStrength, 0.0, 1.0), RAIN_SKY_CURVE);
		float cloudLumScaleMin = mix(0.5, 0.06, cloudRainT);
		cloudLumScale = clamp(cloudLumScale, cloudLumScaleMin, 4.0);
		cloudSkyColor *= cloudLumScale;
		
		// blend sky behind cloud
		float skyBlend = 1.0 - aHere;
		outColor = outColor + cloudSkyColor * skyBlend;
	}

	// debug: visualize classification
	//#define DEBUG_CLASSIFICATION
	#ifdef DEBUG_CLASSIFICATION
		float debugDepth = clamp(depthLinear / far, 0.0, 1.0);
		if (land < 0.5) outColor = vec3(0.0, debugDepth, 1.0);
		else outColor = vec3(1.0, debugDepth, 0.0);
	#endif

	gl_FragData[0] = colorSample;
	gl_FragData[1] = depthSample;
	vec4 rgmb = encodeRGBM(outColor);
	gl_FragData[3] = vec4(rgmb.rgb, 1.0);
	// gaux1: r=motion blur mask, g=RGBM mult, b=land, a=smoothness
	gl_FragData[4] = vec4(noblur, rgmb.a, land, aux.a);
}
