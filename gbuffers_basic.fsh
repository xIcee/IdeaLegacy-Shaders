#version 120

varying vec4 color;

const int GL_LINEAR = 9729;
const int GL_EXP = 2048;

uniform int fogMode;


void main() {
	gl_FragData[0] = color;
	gl_FragData[1] = vec4(vec3(gl_FragCoord.z), 1.0);
	
	// gaux1: mark as non-terrain (b=0.5 indicates basic geometry)
	gl_FragData[4] = vec4(0.0, 0.0, 0.5, 1.0);
}