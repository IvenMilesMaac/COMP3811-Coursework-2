// You will need to define your own tests. Refer to CW1 or Exercise G.3 for
// examples.

#include <catch2/catch_amalgamated.hpp>
#include "../vmlib/mat44.hpp"
#include "helpers.hpp"

static const Mat44f Zero = { 0.f };
static const Mat44f A = { {
    2.f, 1.f, 5.f, 2.f,
    0.f, 6.f, 7.f, 3.f,
    3.f, 1.f, 4.f, 5.f,
    2.f, 1.f, 0.f, 1.f
} };

TEST_CASE("Matrix multiplication", "[Mat44f][operator]")
{
    Mat44f B = { {
        1.f, 2.f, 0.f, 1.f,
        3.f, 0.f, 2.f, 1.f,
        4.f, 1.f, 3.f, 0.f,
        2.f, 5.f, 1.f, 4.f
    } };

    SECTION("Multiplying with identity matrix") {
        Mat44f I = kIdentity44f;
        REQUIRE(isEqual((A * I), A));
        REQUIRE(isEqual((I * A), A));
    }

    SECTION("Multiplying with zero matrix") {
        REQUIRE(isEqual((A * Zero), Zero));
        REQUIRE(isEqual((Zero * A), Zero));
    }

    SECTION("Multiplication with expected result") {
        Mat44f Expected = { {
            29.f, 19.f, 19.f, 11.f,
            52.f, 22.f, 36.f, 18.f,
            32.f, 35.f, 19.f, 24.f,
            7.f, 9.f, 3.f, 7.f
        } };
        REQUIRE(isEqual((A * B), Expected));
    }
}

TEST_CASE("Matrix-vector multiplication", "[Mat44f][Vec4f][operator]")
{
    Vec4f V = { 1.f, 3.f, 7.f, 4.f };
    SECTION("Multiplying with identity matrix") {
        Mat44f I = kIdentity44f;
        // implemented function in Mat44f doesn't account for other direction
        REQUIRE(isEqual((I * V), V));
    }

    SECTION("Multiplying with zero matrix") {
        Vec4f Expected = { 0.f, 0.f, 0.f, 0.f };
        REQUIRE(isEqual((Zero * V), Expected));
    }

    SECTION("Multiplication with expected result") {
        Vec4f Expected = { 48.f, 79.f, 54.f, 9.f };
        REQUIRE(isEqual((A * V), Expected));
    }
}