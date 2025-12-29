//
#version 130

// bump mapping with parallax occlusion
// requires terrain_nh.png

#define ENABLE_PARALLAX_OCCLUSION

#define INTESTINE
#define SOULSAND
#define PORTAL


#ifdef ENABLE_PARALLAX_OCCLUSION
const float HEIGHT_MULT = 1.25; // bump map height multiplier
const float PARALLAX_OCCLUSION_SHADOW_MULT = .75; // shadow intensity in bump valleys (0=none, 1=full)

#endif


#ifdef _ENABLE_GL_TEXTURE_2D

uniform sampler2D sampler0;
centroid in vec4 texCoord;

const float SQUIRM_DISTANCE = 30.0;

#ifdef _ENABLE_BUMP_MAPPING

uniform sampler2D sampler1;
uniform sampler2D sampler2;
uniform sampler2D envMap;

vec3 intervalMult;

in vec4 specMultiplier;
in float useCelestialSpecularMapping;
in vec3 lightVector;
in vec3 viewVector;
in vec3 viewVectorSpec;
in vec3 normal;

const float MAX_DISTANCE = 50.0;
const int MAX_POINTS = 50;

#ifdef ENABLE_PARALLAX_OCCLUSION
#ifdef ENABLE_PARALLAX_SELFSHADOW
const int SELFSHADOW_LOOP = 50;
const float SELFSHADOW_ROUGHNESS = 2.0;
#endif
#endif

#endif // ENABLE_GL_BUMP_MAPPING
#endif // ENABLE_GL_TEXTURE_2D

uniform float near;
uniform float far;
uniform int worldTime;
uniform int renderType;
uniform int fogMode;

in float FogFragCoord;
in float distance;
in float texID;
in float blockID;

struct lightSource
{
    int itemId;
    float magnitude;
    vec4 specular;
};

uniform lightSource heldLight;

const int GL_LINEAR = 9729;
const int GL_EXP = 2048;

const float PI = 3.1415926535897932384626433832795;
const float PI2 = 6.283185307179586476925286766559;

in vec4 vertColor;

vec3
getAmbient ()
{

    vec3 tint = vec3 (1.0, 1.0, 1.0);
    float time;
    if (worldTime >= 22000.0)
    {                           // sunrise!
        time = min (max ((23000.0 - worldTime) / 1000.0, -1.0), 1.0);
        if (time > 0)
        {                       // Sky starts to colorize
            tint[0] = 0.95 + (0.15 * (1.0 - time)); // .95 to 1.1
            tint[1] = 0.95 - (0.05 * (1.0 - time)); // .95 to .9
            tint[2] = 1.1 - (0.1 * (1.0 - time));   // 1.1 to 1.0
        }
        else
        {                       // Sun rises
            tint[0] = 1.1 - (0.05 * (abs (time)));  // 1.1 to 1.05
            tint[1] = 0.9 + (0.0 * (abs (time)));   // .9 to .9
            tint[2] = 1.0 + (0.05 * (abs (time)));  // 1.0 to 1.05
        }
    }
    else if (worldTime >= 12000)
    {                           // night time !
        time = min (max ((13000.0 - worldTime) / 1000.0, -1.0), 1.0);
        if (time > 0)
        {                       // Sunset
            tint[0] = 1.1 + (0.15 * (1.0 - time));  // 1.1 to 1.25
            tint[1] = 1.0 - (0.05 * (1.0 - time));  // 1.0 to 0.95
            tint[2] = 0.9 - (0.15 * (1.0 - time));  // .9 to .75
        }
        else
        {                       // Night begins
            tint[0] = 1.25 - (0.3 * (abs (time)));  // 1.25 to .95
            tint[1] = 0.95 - (0.0 * (abs (time)));  // .95 to .95
            tint[2] = 0.75 + (0.35 * (abs (time))); // .65 to 1.1
        }
    }
    else if (worldTime >= 0)
    {                           // Day time, from dusk til dawn!
        time = min (max ((6000.0 - worldTime) / 6000.0, -1.0), 1.0);
        if (time > 0)
        {                       // Morning
            tint[0] = 1.05 - (0.05 * (1.0 - time)); // 1.05 to 1
            tint[1] = 0.9 + (0.1 * (1.0 - time));   // .9 to 1
            tint[2] = 1.05 - (0.05 * (1.0 - time)); // 1.05 to 1
        }
        else
        {                       // Afternoon
            tint[0] = 1 + (0.1 * (abs (time))); // 1 to 1.1
            tint[1] = 1 - (0.0 * (abs (time))); // 1 to 1
            tint[2] = 1 - (0.1 * (abs (time))); // 1 to .9
        }
    }
    return tint;
}

void
main ()
{

#ifdef _ENABLE_GL_TEXTURE_2D

    vec3 coord = vec3 (texCoord.st, 1.0);
    vec4 vC = vec4 (0.0, 0.0, 0.0, 0.0);
    vec4 baseColor;

#ifdef INTESTINE
    if (round (texID) == 103.0 && distance <= SQUIRM_DISTANCE)
    {
        float t =
            (24000.0 - (2.0 * abs (12000.0 - float (worldTime)))) / 250.0;
        vec3 offset, base;
        coord = modf (16.0 * coord, base);
        offset =
            vec3 (cos (PI2 * coord.s) * cos (PI2 * (coord.t + 2.0 * t)) *
                  cos (PI2 * t) / 40.0,
                  -cos (PI2 * (coord.s + t)) * sin (2.0 * PI2 * coord.t) /
                  40.0, 0);

        coord = mod (coord + offset, vec3 (1.0)) + base;
        coord = coord / 16;
    }
#endif

#ifdef SOULSAND
    if (round (texID) == 104.0 && distance <= SQUIRM_DISTANCE)
    {
        float t =
            (24000.0 - (2.0 * abs (12000.0 - float (worldTime)))) / 1000.0;
        vec3 offset, base;
        coord = modf (16.0 * coord, base);
        offset =
            vec3 (cos (PI2 * coord.s) * cos (PI2 * (coord.t + 2.0 * t)) *
                  cos (PI2 * t) / 24.0,
                  -cos (PI2 * (coord.s + t)) * sin (2.0 * PI2 * coord.t) /
                  32.0, 0);

        coord = mod (coord + offset, vec3 (1.0)) + base;
        coord = coord / 16;
    }
#endif

#ifdef PORTAL
    if (round (texID) == 14.0 && distance <= SQUIRM_DISTANCE)
    {
        float t =
            (24000.0 - (2.0 * abs (12000.0 - float (worldTime)))) / 200.0;
        vec3 offset, base;
        coord = modf (16.0 * coord, base);
        offset =
            vec3 (cos (PI2 * coord.s) * cos (PI2 * (coord.t + 2.0 * t)) *
                  cos (PI2 * t) / 6.0,
                  -cos (PI2 * (coord.s + t)) * sin (2.0 * PI2 * coord.t) /
                  18.0, 0);

        coord = mod (coord + offset, vec3 (1.0)) + base;
        coord = coord / 16;

        vC = texture2D (sampler0, coord.st);
        float lum = vC[0] * 0.299 + vC[1] * 0.587 + vC[2] * 0.114;
        lum += clamp (sin ((3.0 * lum + 5.0 * t)) + 0.5, 0.5, 3.0);
        vC = vec4 (vC.rgb * lum, -0.2);
    }
#endif

#ifdef _ENABLE_BUMP_MAPPING

    if (distance <= MAX_DISTANCE)
    {

        vec3 N = normalize (normal);
        vec3 V = normalize (viewVectorSpec);
        vec3 L = normalize (lightVector);

        int tSize = 0;
        tSize = textureSize (sampler1, 0).x;
        float multTest = 1.0;
        if (tSize > 16)
        {

            if (tSize >= 4096)
            {
                intervalMult = vec3 (0.00008828125, 0.00008828125, 0.03);
            }
            else if (tSize >= 2048)
            {
                intervalMult = vec3 (0.00048828125, 0.00048828125, 0.145);
            }
            else if (tSize >= 1024)
            {
                intervalMult = vec3 (0.0009, 0.0009, 0.25);
            }
            else if (tSize >= 512)
            {
                intervalMult = vec3 (0.0019, 0.0019, 0.5);
            }
            else if (tSize >= 256)
            {
                intervalMult = vec3 (0.0039, 0.0039, 4.5);
            }
            else
            {
                intervalMult = vec3 (0.00008828125, 0.00008828125, 0.03);
            }

#ifdef ENABLE_PARALLAX_OCCLUSION
            intervalMult[2] /= HEIGHT_MULT;
            if (texture2D (sampler1, coord.st).a < 1.0)
            {
                vec2 minCoord =
                    vec2 (texCoord.s - mod (texCoord.s, 0.0625),
                          texCoord.t - mod (texCoord.t, 0.0625));
                vec2 maxCoord =
                    vec2 (minCoord.s + 0.0625, minCoord.t + 0.0625);

                vec3 vt_delta = viewVector * intervalMult;

                for (int loopCount = 0;
                     texture2D (sampler1, coord.st).a < coord.z
                     && loopCount < MAX_POINTS; ++loopCount)
                {
                    coord += vt_delta;
                    //boundary check and wrap correction
                    if (coord.s < minCoord.s)
                    {
                        coord.s += 0.0625;
                    }
                    else if (coord.s >= maxCoord.s)
                    {
                        coord.s -= 0.0625;
                    }
                    if (coord.t < minCoord.t)
                    {
                        coord.t += 0.0625;
                    }
                    else if (coord.t >= maxCoord.t)
                    {
                        coord.t -= 0.0625;
                    }
                }

            }
            multTest =
                clamp ((pow
                        ((texture2D (sampler1, coord.st).a + 1.2) * 0.6,
                         3) / (0.01 + PARALLAX_OCCLUSION_SHADOW_MULT)), 0.02,
                       1.0);
#endif
        }

        baseColor = texture2D (sampler0, coord.st) * vertColor * multTest;

    }
    else
    {
        baseColor = texture2D (sampler0, coord.st) * vertColor;
    }
#else
    baseColor = texture2D (sampler0, coord.st) * vertColor;
#endif // _ENABLE_BUMP_MAPPING
    gl_FragColor = baseColor + vC;
#else // ENABLE_GL_TEXTURE_2D
    gl_FragColor = vertColor;
#endif // ENABLE_GL_TEXTURE_2D
}
