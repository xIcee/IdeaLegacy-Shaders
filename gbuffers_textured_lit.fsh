#version 120

uniform sampler2D texture;
uniform sampler2D lightmap;

uniform mat4 gbufferProjectionInverse;
uniform mat4 gbufferPreviousProjection;

uniform mat4 gbufferModelViewInverse;
uniform mat4 gbufferPreviousModelView;

uniform vec3 cameraPosition;
uniform vec3 previousCameraPosition;

varying vec4 color;
varying vec4 texcoord;
varying vec4 lmcoord;
varying vec2 rawLightmap; // Raw light values: x = block light, y = sky light (0-1)

const int GL_LINEAR = 9729;
const int GL_EXP = 2048;

uniform int fogMode;

float pbrLuma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

float pbrSaturation(vec3 c) {
	float mx = max(c.r, max(c.g, c.b));
	float mn = min(c.r, min(c.g, c.b));
	return (mx > 0.0) ? (mx - mn) / mx : 0.0;
}

float pbrGetDif(float lOriginal, vec2 offsetCoord) {
	const float normalThreshold = 0.05;
	const float normalClamp = 0.2;
	
	float lNearby = length(texture2D(texture, offsetCoord).rgb);
	float dif = lOriginal - lNearby;
	
	if (dif > 0.0) dif = max(dif - normalThreshold, 0.0);
	else           dif = min(dif + normalThreshold, 0.0);
	
	return clamp(dif, -normalClamp, normalClamp);
}

float pbrTextureDetail(vec2 uv, vec3 centerColor) {
	vec2 texelStep = fwidth(uv) / 16.0;
	
	float lOriginal = length(centerColor);
	
	float difU = pbrGetDif(lOriginal, uv + vec2(0.0, texelStep.y));
	float difD = pbrGetDif(lOriginal, uv - vec2(0.0, texelStep.y));
	float difR = pbrGetDif(lOriginal, uv + vec2(texelStep.x, 0.0));
	float difL = pbrGetDif(lOriginal, uv - vec2(texelStep.x, 0.0));
	
	return abs(difU) + abs(difD) + abs(difR) + abs(difL);
}

float pbrLiteSmoothness(vec4 texSample, vec2 uv) {
	vec3 c = texSample.rgb;
	float luma = pbrLuma(c);
	float sat = pbrSaturation(c);
	float detail = pbrTextureDetail(uv, c);
	
	float smooth = 0.0;
	
	float grayness = 1.0 - sat;
	float luma2 = luma * luma;
	smooth += luma2 * luma2 * grayness * 0.5; // pow(luma,4) optimized
	smooth -= sat * 0.15;
	smooth -= detail * 1.5;
	
	float lowDetail = max(0.0, 0.1 - detail);
	smooth += lowDetail * luma * 2.0;
	
	smooth *= smoothstep(0.04, 0.25, luma);
	
	return clamp(smooth, 0.0, 1.0);
}

void main() {
	vec4 baseTex = texture2D(texture, texcoord.st);
	vec4 lm = texture2D(lightmap, lmcoord.st);
	float smoothness = pbrLiteSmoothness(baseTex, texcoord.st);

	gl_FragData[0] = baseTex * lm * (color);
	// keep depth alpha stable; rawLightmap.* is often broken in legacy pipelines
	gl_FragData[1] = vec4(vec3(gl_FragCoord.z), 1.0);
	// gnormal.r: lighting luminance proxy (legacy-safe)
	vec3 litColor = (baseTex.rgb * lm.rgb) * color.rgb;
	float litLum = dot(litColor, vec3(0.2126, 0.7152, 0.0722));
	float texLum = dot(baseTex.rgb, vec3(0.2126, 0.7152, 0.0722));
	float lightProxy = clamp(litLum / max(texLum, 0.04), 0.0, 1.0);
	gl_FragData[2] = vec4(lightProxy, 0.0, 0.0, 1.0);

	// gaux1: r=highlightMult, g=metallic, b=land, a=smoothness
	gl_FragData[4] = vec4(0.07, 0.0, 1.0, smoothness);

	//gl_FragData[1] = vec4(0.0);

		
		
	// fog applied in composite.fsh
}