#ifndef MAT44_HPP_E7187A26_469E_48AD_A3D2_63150F05A4CA
#define MAT44_HPP_E7187A26_469E_48AD_A3D2_63150F05A4CA
// SOLUTION_TAGS: gl-(ex-[^12]|cw-2|resit)

#include <cmath>
#include <cassert>
#include <cstdlib>

#include "vec3.hpp"
#include "vec4.hpp"

/** Mat44f: 4x4 matrix with floats
 *
 * See vec2f.hpp for discussion. Similar to the implementation, the Mat44f is
 * intentionally kept simple and somewhat bare bones.
 *
 * The matrix is stored in row-major order (careful when passing it to OpenGL).
 *
 * The overloaded operator [] allows access to individual elements. Example:
 *    Mat44f m = ...;
 *    float m12 = m[1,2];
 *    m[0,3] = 3.f;
 *
 * (Multi-dimensionsal subscripts in operator[] is a C++23 feature!)
 *
 * The matrix is arranged as:
 *
 *   ⎛ 0,0  0,1  0,2  0,3 ⎞
 *   ⎜ 1,0  1,1  1,2  1,3 ⎟
 *   ⎜ 2,0  2,1  2,2  2,3 ⎟
 *   ⎝ 3,0  3,1  3,2  3,3 ⎠
 */
struct Mat44f
{
	float v[16];

	constexpr
	float& operator[] (std::size_t aI, std::size_t aJ) noexcept
	{
		assert( aI < 4 && aJ < 4 );
		return v[aI*4 + aJ];
	}
	constexpr
	float const& operator[] (std::size_t aI, std::size_t aJ) const noexcept
	{
		assert( aI < 4 && aJ < 4 );
		return v[aI*4 + aJ];
	}
};

// Identity matrix
constexpr Mat44f kIdentity44f = { {
	1.f, 0.f, 0.f, 0.f,
	0.f, 1.f, 0.f, 0.f,
	0.f, 0.f, 1.f, 0.f,
	0.f, 0.f, 0.f, 1.f
} };

// Common operators for Mat44f.
// Note that you will need to implement these yourself.

constexpr
Mat44f operator*( Mat44f const& aLeft, Mat44f const& aRight ) noexcept
{
	Mat44f result = { 0.f };

	// Standard Matrix Multiplication: Row of Left * Column of Right
	for (std::size_t r = 0; r < 4; ++r)
	{
		for (std::size_t c = 0; c < 4; ++c)
		{
			float sum = 0.f;
			for (std::size_t k = 0; k < 4; ++k)
			{
				sum += aLeft[r, k] * aRight[k, c];
			}
			result[r, c] = sum;
		}
	}
	return result;
}

constexpr
Vec4f operator*( Mat44f const& aLeft, Vec4f const& aRight ) noexcept
{
	return Vec4f{
		aLeft[0,0] * aRight.x + aLeft[0,1] * aRight.y + aLeft[0,2] * aRight.z + aLeft[0,3] * aRight.w,
		aLeft[1,0] * aRight.x + aLeft[1,1] * aRight.y + aLeft[1,2] * aRight.z + aLeft[1,3] * aRight.w,
		aLeft[2,0] * aRight.x + aLeft[2,1] * aRight.y + aLeft[2,2] * aRight.z + aLeft[2,3] * aRight.w,
		aLeft[3,0] * aRight.x + aLeft[3,1] * aRight.y + aLeft[3,2] * aRight.z + aLeft[3,3] * aRight.w
	};
}

// Functions:

Mat44f invert( Mat44f const& aM ) noexcept;

inline
Mat44f transpose( Mat44f const& aM ) noexcept
{
	Mat44f ret;
	for( std::size_t i = 0; i < 4; ++i )
	{
		for( std::size_t j = 0; j < 4; ++j )
			ret[j,i] = aM[i,j];
	}
	return ret;
}

inline
Mat44f make_rotation_x( float aAngle ) noexcept
{
	Mat44f m = kIdentity44f;
	float c = std::cos(aAngle);
	float s = std::sin(aAngle);

	// Rotate Y and Z axes (Row 1 and Row 2)
	m[1, 1] = c;
	m[1, 2] = -s;
	m[2, 1] = s;
	m[2, 2] = c;

	return m;
}


inline
Mat44f make_rotation_y( float aAngle ) noexcept
{
	Mat44f m = kIdentity44f;
	float c = std::cos(aAngle);
	float s = std::sin(aAngle);

	// Rotate X and Z axes (Row 0 and Row 2)
	m[0, 0] = c;
	m[0, 2] = s;
	m[2, 0] = -s;
	m[2, 2] = c;

	return m;
}

inline
Mat44f make_rotation_z( float aAngle ) noexcept
{
	Mat44f m = kIdentity44f;
	float c = std::cos(aAngle);
	float s = std::sin(aAngle);

	// Rotate X and Y axes (Row 0 and Row 1)
	m[0, 0] = c;
	m[0, 1] = -s;
	m[1, 0] = s;
	m[1, 1] = c;

	return m;
}

inline
Mat44f make_translation( Vec3f aTranslation ) noexcept
{
	Mat44f m = kIdentity44f;

	// Translation goes in the last column (Column 3)
	m[0, 3] = aTranslation.x;
	m[1, 3] = aTranslation.y;
	m[2, 3] = aTranslation.z;

	return m;
}
inline
Mat44f make_scaling( float aSX, float aSY, float aSZ ) noexcept
{
	Mat44f m = kIdentity44f;

	// Scale factor is along the diagonal
	m[0, 0] = aSX;
	m[1, 1] = aSY;
	m[2, 2] = aSZ;

	return m;
}

inline
Mat44f make_perspective_projection( float aFovInRadians, float aAspect, float aNear, float aFar ) noexcept
{

	Mat44f m = { 0.f };

	float tanHalfFov = std::tan(aFovInRadians / 2.f);
	float s = 1.f / tanHalfFov;

	// Scale X and Y based on FOV and Aspect Ratio
	m[0, 0] = s / aAspect;
	m[1, 1] = s;

	// Remap Z to [-1, 1] range
	m[2, 2] = -(aFar + aNear) / (aFar - aNear);
	m[2, 3] = -(2.f * aFar * aNear) / (aFar - aNear);

	// Copy -Z into W for perspective division
	m[3, 2] = -1.f;

	return m;
}

// Added fuction for camera view matrix
inline
Mat44f construct_camera_view(Vec3f const& forward, Vec3f const& up, Vec3f const& right, Vec3f const& position)
{
	Mat44f view = Mat44f{ {
		right.x,	up.x,		-forward.x,		-dot(right, position),
		right.y,	up.y,		-forward.y,		-dot(up, position),
		right.z,	up.z,		-forward.z,		dot(forward, position),
		0.f,		0.f,        0.f,			1.f
	} };

	return view;
};

#endif // MAT44_HPP_E7187A26_469E_48AD_A3D2_63150F05A4CA
