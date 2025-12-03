#ifndef HELPERS_HPP
#define HELPERS_HPP

#include "../vmlib/mat44.hpp"
#include <cmath>

// compare 4x4 matrices
inline bool isEqual(const Mat44f& a, const Mat44f& b, float eps = 1e-5f) {
	for (std::size_t r = 0; r < 4; ++r) {
		for (std::size_t c = 0; c < 4; ++c) {
			if (std::fabs(a[r, c] - b[r, c]) > eps) {
				return false;
			}
		}
	}
	return true;
}

// compare 4x1 vectors
inline bool isEqual(const Vec4f& a, const Vec4f& b, float eps = 1e-5f) {
	return	std::fabs(a.x - b.x) < eps &&
		std::fabs(a.y - b.y) < eps &&
		std::fabs(a.z - b.z) < eps &&
		std::fabs(a.w - b.w) < eps;
}

#endif // HELPERS_HPP