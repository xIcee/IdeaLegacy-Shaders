#version 120

uniform sampler2D texture;
uniform sampler2D lightmap;

varying vec4 color;
varying vec4 texcoord;
varying vec4 lmcoord;

void main() {
	vec4 lm = texture2D(lightmap, lmcoord.st);
	// minimum ambient so caves aren't pitch black
	lm.rgb = max(lm.rgb, vec3(0.030, 0.022, 0.060));
	gl_FragData[0] = texture2D(texture, texcoord.st) * lm * color;
	gl_FragData[1] = vec4(vec3(gl_FragCoord.z), 1.0);
	// gaux1: r=motion blur, g=weather lm luminance, b=land, a=overlay flag
	float lmLum = dot(lm.rgb, vec3(0.3333));
	gl_FragData[4] = vec4(0.0, lmLum, 1.0, 0.0);
		
}