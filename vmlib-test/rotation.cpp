// You will need to define your own tests. Refer to CW1 or Exercise G.3 for
// examples.

#include <catch2/catch_amalgamated.hpp>
#include "../vmlib/mat44.hpp"
#include "helpers.hpp"
#include <numbers>


TEST_CASE("Rotation of 0 returns identity", "[rotate][identity]") {
	float angle = 0.0f;
	SECTION("Rotation around X axis") {
        Mat44f Rx = make_rotation_x(angle);
        REQUIRE(isEqual(Rx, kIdentity44f));
	}
    SECTION("Rotation around Y axis") {
		Mat44f Ry = make_rotation_y(angle);
		REQUIRE(isEqual(Ry, kIdentity44f));
    }
    SECTION("Rotation around Z axis") {
        Mat44f Rz = make_rotation_z(angle);
        REQUIRE(isEqual(Rz, kIdentity44f));
    }
}

TEST_CASE("Rotation of 90 degrees", "[rotate][expected]") {
	float angle = std::numbers::pi_v<float> / 2.0f;
    SECTION("Rotation around X axis") {
        Mat44f Rx = make_rotation_x(angle);
        Mat44f Expected = { {
            1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, -1.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 1.f
        } };
        REQUIRE(isEqual(Rx, Expected));
	}
    SECTION("Rotation around Y axis") {
        Mat44f Ry = make_rotation_y(angle);
        Mat44f Expected = { {
            0.f, 0.f, 1.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            -1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 1.f
        } };
        REQUIRE(isEqual(Ry, Expected));
    }
    SECTION("Rotation around Z axis") {
        Mat44f Rz = make_rotation_z(angle);
        Mat44f Expected = { {
            0.f, -1.f, 0.f, 0.f,
            1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f
        } };
        REQUIRE(isEqual(Rz, Expected));
	}
}

TEST_CASE("Rotation of 180 degrees", "[rotate][expected]") {
	float angle = std::numbers::pi_v<float>;
    SECTION("Rotation around X axis") {
		Mat44f Rx = make_rotation_x(angle);
		Mat44f Expected = { {
			1.f, 0.f, 0.f, 0.f,
			0.f, -1.f, 0.f, 0.f,
			0.f, 0.f, -1.f, 0.f,
			0.f, 0.f, 0.f, 1.f
		} };
		REQUIRE(isEqual(Rx, Expected));
    }
	SECTION("Rotation around Y axis") {
        Mat44f Ry = make_rotation_y(angle);
        Mat44f Expected = { {
            -1.f, 0.f, 0.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            0.f, 0.f, -1.f, 0.f,
            0.f, 0.f, 0.f, 1.f
        } };
        REQUIRE(isEqual(Ry, Expected));
    }
    SECTION("Rotation around Z axis") {
        Mat44f Rz = make_rotation_z(angle);
        Mat44f Expected = { {
            -1.f, 0.f, 0.f, 0.f,
            0.f, -1.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f
        } };
        REQUIRE(isEqual(Rz, Expected));
    }
}

TEST_CASE("Rotation of 270 degrees", "[rotate][expected]") {
    float angle = 1.5f * std::numbers::pi_v<float>;
    SECTION("Rotation around X axis") {
        Mat44f Rx = make_rotation_x(angle);
        Mat44f Expected = { {
            1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, -1.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 1.f
        } };
        REQUIRE(isEqual(Rx, Expected));
    }
    SECTION("Rotation around Y axis") {
        Mat44f Ry = make_rotation_y(angle);
        Mat44f Expected = { {
            0.f, 0.f, -1.f, 0.f,
            0.f, 1.f, 0.f, 0.f,
            1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 0.f, 1.f
        } };
        REQUIRE(isEqual(Ry, Expected));
    }
    SECTION("Rotation around Z axis") {
        Mat44f Rz = make_rotation_z(angle);
        Mat44f Expected = { {
            0.f, 1.f, 0.f, 0.f,
            -1.f, 0.f, 0.f, 0.f,
            0.f, 0.f, 1.f, 0.f,
            0.f, 0.f, 0.f, 1.f
        } };
        REQUIRE(isEqual(Rz, Expected));
    }
}

TEST_CASE("Rotation of 360 returns identity", "[rotate][identity]") {
    float angle = 2.0f * std::numbers::pi_v<float>;
    SECTION("Rotation around X axis") {
        Mat44f Rx = make_rotation_x(angle);
        REQUIRE(isEqual(Rx, kIdentity44f));
    }
    SECTION("Rotation around Y axis") {
        Mat44f Ry = make_rotation_y(angle);
        REQUIRE(isEqual(Ry, kIdentity44f));
    }
    SECTION("Rotation around Z axis") {
        Mat44f Rz = make_rotation_z(angle);
        REQUIRE(isEqual(Rz, kIdentity44f));
    }
}