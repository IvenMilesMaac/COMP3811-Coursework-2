// You will need to define your own tests. Refer to CW1 or Exercise G.3 for
// examples.

#include <catch2/catch_amalgamated.hpp>
#include "../vmlib/mat44.hpp"
#include "helpers.hpp"

TEST_CASE("Translation with 3x1 vector", "[translate]") {

	SECTION("Translation of 0 returns identity") {
		Vec3f T = { 0.f, 0.f, 0.f };
		Mat44f result = make_translation(T);
		REQUIRE(isEqual(result, kIdentity44f));
	}

	SECTION("Translation with expected result") {
		Vec3f T = { 2.f, 3.f, 5.f };
		Mat44f result = make_translation(T);
		Mat44f Expected = { {
			1.f, 0.f, 0.f, 2.f,
			0.f, 1.f, 0.f, 3.f,
			0.f, 0.f, 1.f, 5.f,
			0.f, 0.f, 0.f, 1.f
		} };
		REQUIRE(isEqual(result, Expected));
	}

	SECTION("Translation in negative values") {
		Vec3f T = { -6.f, 0.f, -1.f };
		Mat44f result = make_translation(T);
		Mat44f Expected = { {
			1.f, 0.f, 0.f, -6.f,
			0.f, 1.f, 0.f, 0.f,
			0.f, 0.f, 1.f, -1.f,
			0.f, 0.f, 0.f, 1.f
		} };
		REQUIRE(isEqual(result, Expected));
	}
}