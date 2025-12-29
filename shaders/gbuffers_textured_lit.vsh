#version 120

varying vec4 color;
varying vec4 texcoord;
varying vec4 lmcoord;
varying vec2 rawLightmap;

void main() {
	gl_Position = ftransform();
	
	color = gl_Color;
	
	texcoord = gl_TextureMatrix[0] * gl_MultiTexCoord0;

	lmcoord = gl_TextureMatrix[1] * gl_MultiTexCoord1;
	
	rawLightmap = gl_MultiTexCoord1.xy / 240.0;

	gl_FogFragCoord = gl_Position.z;
}