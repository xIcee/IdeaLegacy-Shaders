#version 120

varying vec4 color;
varying vec4 texcoord;
varying vec4 lmcoord;
varying vec2 rawLightmap; // Raw light values: x = block light, y = sky light (0-1)
varying vec4 bloommask;

attribute vec4 mc_Entity;

uniform int worldTime;
uniform float rainStrength;

// NOTE: Waving animation system removed - was dormant/disabled code.
// If waving foliage is desired, implement a cleanroom solution using:
//   - Simplex/Perlin noise for organic motion
//   - Wind direction uniform for coherent movement
//   - Vertex displacement based on world position + time

void main() {

	texcoord = gl_MultiTexCoord0;
	
	vec4 position = gl_Vertex;
	
	bloommask = vec4(0.0);
	
	

	gl_Position = gl_ProjectionMatrix * (gl_ModelViewMatrix * position);
	
	color = gl_Color;
	
	lmcoord = gl_TextureMatrix[1] * gl_MultiTexCoord1;
	
	// raw lightmap values (0-1)
	rawLightmap = gl_MultiTexCoord1.xy / 240.0;
	
	gl_FogFragCoord = gl_Position.z;
}