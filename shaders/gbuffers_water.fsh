#version 120

uniform sampler2D texture;
uniform sampler2D lightmap;

varying vec4 color;
varying vec4 texcoord;
varying vec4 lmcoord;
varying vec2 rawLightmap;  // x = block light, y = sky light (0-1)

const int GL_LINEAR = 9729;
const int GL_EXP = 2048;

uniform int fogMode;


// ============================================================================
//  SMOOTHNESS ESTIMATION
// ============================================================================

// Perceptual luminance
float pbrLuma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

// Color saturation (HSV-style)
float pbrSaturation(vec3 c) {
	float mx = max(c.r, max(c.g, c.b));
	float mn = min(c.r, min(c.g, c.b));
	return (mx > 0.0) ? (mx - mn) / mx : 0.0;
}

// Sample brightness difference from neighbor texel
float pbrGetDif(float lOriginal, vec2 offsetCoord) {
	const float normalThreshold = 0.05;
	const float normalClamp = 0.2;
	
	float lNearby = length(texture2D(texture, offsetCoord).rgb);
	float dif = lOriginal - lNearby;
	
	if (dif > 0.0) dif = max(dif - normalThreshold, 0.0);
	else           dif = min(dif + normalThreshold, 0.0);
	
	return clamp(dif, -normalClamp, normalClamp);
}

// Measure local texture variation (simplified version for water)
float pbrTextureDetail(vec2 uv, vec3 centerColor) {
	vec2 texelStep = fwidth(uv) / 16.0;
	
	float lOriginal = length(centerColor);
	
	float difU = pbrGetDif(lOriginal, uv + vec2(0.0, texelStep.y));
	float difD = pbrGetDif(lOriginal, uv - vec2(0.0, texelStep.y));
	float difR = pbrGetDif(lOriginal, uv + vec2(texelStep.x, 0.0));
	float difL = pbrGetDif(lOriginal, uv - vec2(texelStep.x, 0.0));
	
	return abs(difU) + abs(difD) + abs(difR) + abs(difL);
}

// Estimate surface smoothness from texture properties.
// Water defaults to high smoothness for reflections.
float pbrLiteSmoothness(vec4 texSample, vec2 uv) {
	vec3 c = texSample.rgb;
	float luma = pbrLuma(c);
	float sat = pbrSaturation(c);
	float detail = pbrTextureDetail(uv, c);
	
	// Base smoothness from brightness
	float smooth = luma * 0.6;
	
	// Bright desaturated surfaces = smoother
	float grayness = 1.0 - sat;
	smooth += luma * luma * grayness * 0.4;
	
	// Saturated colors = rougher
	smooth -= sat * 0.2;
	
	// High texture detail = rougher
	smooth -= detail * 1.0;
	
	// Low texture detail = smoother
	float lowDetail = max(0.0, 0.15 - detail);
	smooth += lowDetail * 1.5;
	
	// Dark surfaces = low smoothness
	smooth *= smoothstep(0.05, 0.2, luma);
	
	return clamp(smooth, 0.0, 1.0);
}


// ============================================================================
//  MAIN
// ============================================================================

void main() {
	vec4 baseTex = texture2D(texture, texcoord.st);
	vec4 lm = texture2D(lightmap, lmcoord.st);
	
	// Water is always smooth (minimum 0.75 for good reflections)
	float smoothness = max(0.75, pbrLiteSmoothness(baseTex, texcoord.st));

	// --- G-Buffer Outputs ---
	gl_FragData[0] = baseTex * lm * color;
	gl_FragData[1] = vec4(vec3(gl_FragCoord.z), 1.0);

	// gnormal.r: lighting luminance proxy (legacy-safe)
	vec3 litColor = (baseTex.rgb * lm.rgb) * color.rgb;
	float litLum = dot(litColor, vec3(0.2126, 0.7152, 0.0722));
	float texLum = dot(baseTex.rgb, vec3(0.2126, 0.7152, 0.0722));
	float lightProxy = clamp(litLum / max(texLum, 0.04), 0.0, 1.0);
	gl_FragData[2] = vec4(lightProxy, 0.0, 0.0, 1.0);

	// gaux1: Material properties with WATER MASK in G channel
	//   R: f0 (0.25 for water)
	//   G: water mask (1.0 = water, used by composite for wave normals)
	//   B: land mask (1.0 = geometry)
	//   A: smoothness
	gl_FragData[4] = vec4(0.25, 1.0, 1.0, smoothness);
}