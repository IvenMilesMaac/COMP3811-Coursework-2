#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <print>
#include <numbers>
#include <typeinfo>
#include <stdexcept>

#include <cstdlib>

#include "../support/error.hpp"
#include "../support/program.hpp"
#include "../support/checkpoint.hpp"
#include "../support/debug_output.hpp"

#include "../vmlib/vec4.hpp"
#include "../vmlib/mat44.hpp"

#include "defaults.hpp"


constexpr float kPi = std::numbers::pi_v<float>;

namespace
{
	constexpr char const* kWindowTitle = "COMP3811 - CW2";

	constexpr float kMovementSpeed = 5.f;
	constexpr float kMouseSens = 0.01f;

	struct State_
	{
		ShaderProgram* prog;

		struct CamCtrl_
		{
			bool cameraActive;
			bool actionForward;
			bool actionBackward;
			bool actionLeft;
			bool actionRight;
			bool actionUp;
			bool actionDown;
			bool actionSpeedUp;
			bool actionSlowDown;

			float phi;
			float theta;

			float lastX;
			float lastY;

			Vec3f position;
		} camControl;
	};
	
	void glfw_callback_error_( int, char const* );

	void glfw_callback_mouse_button_(GLFWwindow*, int, int, int);
	void glfw_callback_key_( GLFWwindow*, int, int, int, int );
	void glfw_callback_motion_(GLFWwindow*, double, double);

	struct GLFWCleanupHelper
	{
		~GLFWCleanupHelper();
	};
	struct GLFWWindowDeleter
	{
		~GLFWWindowDeleter();
		GLFWwindow* window;
	};

}

int main() try
{
	// Initialize GLFW
	if( GLFW_TRUE != glfwInit() )
	{
		char const* msg = nullptr;
		int ecode = glfwGetError( &msg );
		throw Error( "glfwInit() failed with '{}' ({})", msg, ecode );
	}

	// Ensure that we call glfwTerminate() at the end of the program.
	GLFWCleanupHelper cleanupHelper;

	// Configure GLFW and create window
	glfwSetErrorCallback( &glfw_callback_error_ );

	glfwWindowHint( GLFW_SRGB_CAPABLE, GLFW_TRUE );
	glfwWindowHint( GLFW_DOUBLEBUFFER, GLFW_TRUE );

	//glfwWindowHint( GLFW_RESIZABLE, GLFW_FALSE );

	glfwWindowHint( GLFW_CONTEXT_VERSION_MAJOR, 4 );
	glfwWindowHint( GLFW_CONTEXT_VERSION_MINOR, 3 );
	glfwWindowHint( GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE );
	glfwWindowHint( GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE );

	glfwWindowHint( GLFW_DEPTH_BITS, 24 );

#	if !defined(NDEBUG)
	// When building in debug mode, request an OpenGL debug context. This
	// enables additional debugging features. However, this can carry extra
	// overheads. We therefore do not do this for release builds.
	glfwWindowHint( GLFW_OPENGL_DEBUG_CONTEXT, GLFW_TRUE );
#	endif // ~ !NDEBUG

	GLFWwindow* window = glfwCreateWindow(
		1280,
		720,
		kWindowTitle,
		nullptr, nullptr
	);

	if( !window )
	{
		char const* msg = nullptr;
		int ecode = glfwGetError( &msg );
		throw Error( "glfwCreateWindow() failed with '{}' ({})", msg, ecode );
	}

	GLFWWindowDeleter windowDeleter{ window };


	// Set up event handling
	State_ state{};
	glfwSetWindowUserPointer(window, &state);

	glfwSetMouseButtonCallback(window, &glfw_callback_mouse_button_);
	glfwSetKeyCallback( window, &glfw_callback_key_ );
	glfwSetCursorPosCallback(window, &glfw_callback_motion_);

	// Set up drawing stuff
	glfwMakeContextCurrent( window );
	glfwSwapInterval( 1 ); // V-Sync is on.

	// Initialize GLAD
	// This will load the OpenGL API. We mustn't make any OpenGL calls before this!
	if( !gladLoadGLLoader( (GLADloadproc)&glfwGetProcAddress ) )
		throw Error( "gladLoadGLLoader() failed - cannot load GL API!" );

	std::print( "RENDERER {}\n", (char const*)glGetString( GL_RENDERER ) );
	std::print( "VENDOR {}\n", (char const*)glGetString( GL_VENDOR ) );
	std::print( "VERSION {}\n", (char const*)glGetString( GL_VERSION ) );
	std::print( "SHADING_LANGUAGE_VERSION {}\n", (char const*)glGetString( GL_SHADING_LANGUAGE_VERSION ) );

	// Ddebug output
#	if !defined(NDEBUG)
	setup_gl_debug_output();
#	endif // ~ !NDEBUG

	// Global GL state
	OGL_CHECKPOINT_ALWAYS();

	// TODO: global GL setup goes here

	OGL_CHECKPOINT_ALWAYS();

	// Get actual framebuffer size.
	// This can be different from the window size, as standard window
	// decorations (title bar, borders, ...) may be included in the window size
	// but not be part of the drawable surface area.
	int iwidth, iheight;
	glfwGetFramebufferSize( window, &iwidth, &iheight );

	glViewport( 0, 0, iwidth, iheight );

	// Other initialization & loading

	// Load shader program
	ShaderProgram prog({
		{ GL_VERTEX_SHADER, "assets/cw2/default.vert" },
		{ GL_FRAGMENT_SHADER, "assets/cw2/default.frag" }
	});
	state.prog = &prog;
	state.camControl.position = Vec3f{ 0.f, -20.f, -80.f };

	// Animation state
	auto last = Clock::now();

	float angle = 0.f;

	OGL_CHECKPOINT_ALWAYS();
	
	// TODO: global GL setup goes here

	OGL_CHECKPOINT_ALWAYS();

	// Main loop
	while( !glfwWindowShouldClose( window ) )
	{
		// Let GLFW process events
		glfwPollEvents();
		
		// Check if window was resized.
		float fbwidth, fbheight;
		{
			int nwidth, nheight;
			glfwGetFramebufferSize( window, &nwidth, &nheight );

			fbwidth = float(nwidth);
			fbheight = float(nheight);

			if( 0 == nwidth || 0 == nheight )
			{
				// Window minimized? Pause until it is unminimized.
				// This is a bit of a hack.
				do
				{
					glfwWaitEvents();
					glfwGetFramebufferSize( window, &nwidth, &nheight );
				} while( 0 == nwidth || 0 == nheight );
			}

			glViewport( 0, 0, nwidth, nheight );
		}

		// Update state
		auto const now = Clock::now();
		float dt = std::chrono::duration_cast<Secondsf>(now - last).count();
		last = now;

		angle += dt * kPi * 0.3f;
		if (angle >= 2.f * kPi)
			angle -= 2.f * kPi;

		// Update camera state
		auto& cam = state.camControl;

		Vec3f forward{
			std::cos( cam.theta ) * std::sin( cam.phi ),
			std::sin( cam.theta ),
			std::cos( cam.theta ) * std::cos( cam.phi )
		};
		forward = normalize(forward);

		Vec3f right = cross(forward, Vec3f{ 0.f, 1.f, 0.f });
		right = normalize(right);

		Vec3f up = cross(right, forward);
		up = normalize(up);

		float speed = kMovementSpeed;
		if (cam.actionSpeedUp)
			speed *= 2.f;
		if (cam.actionSlowDown)
			speed *= 0.5f;

		float dtSpeed = speed * dt;
		if (cam.actionForward)
			cam.position += dtSpeed * forward;
		if (cam.actionBackward)
			cam.position -= dtSpeed * forward;
		if (cam.actionRight)
			cam.position += dtSpeed * right;
		if (cam.actionLeft)
			cam.position -= dtSpeed * right;
		if (cam.actionUp)
			cam.position += dtSpeed * up;
		if (cam.actionDown)
			cam.position -= dtSpeed * up;

		// Draw scene
		OGL_CHECKPOINT_DEBUG();

		// Compute matrices
		Mat44f camera_view = construct_camera_view(forward, up, right, cam.position);
		Mat44f projection = make_perspective_projection(
			60.f * std::numbers::pi_v<float> / 180.f,
			fbwidth / float(fbheight),
			0.1f, 1000.0f
		);
		Mat44f model = kIdentity44f;
		Mat44f projCameraWorld = projection * camera_view * model;

		// Draw frame
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glUseProgram(prog.programId());
		

		OGL_CHECKPOINT_DEBUG();

		// Display results
		glfwSwapBuffers( window );
	}

	// Cleanup.
	//TODO: additional cleanup
	
	return 0;
}
catch( std::exception const& eErr )
{
	std::print( stderr, "Top-level Exception ({}):\n", typeid(eErr).name() );
	std::print( stderr, "{}\n", eErr.what() );
	std::print( stderr, "Bye.\n" );
	return 1;
}


namespace
{
	void glfw_callback_error_( int aErrNum, char const* aErrDesc )
	{
		std::print( stderr, "GLFW error: {} ({})\n", aErrDesc, aErrNum );
	}

	void glfw_callback_mouse_button_(GLFWwindow* aWindow, int aButton, int aAction, int mods)
	{
		// activate / deactivate camera control
		if (GLFW_MOUSE_BUTTON_RIGHT == aButton && GLFW_PRESS == aAction)
		{
			auto* state = static_cast<State_*>(glfwGetWindowUserPointer(aWindow));
			if (state)
			{
				state->camControl.cameraActive = !state->camControl.cameraActive;

				if (state->camControl.cameraActive)
					glfwSetInputMode(aWindow, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
				else
					glfwSetInputMode(aWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}
	}

	void glfw_callback_key_( GLFWwindow* aWindow, int aKey, int, int aAction, int )
	{
		if( GLFW_KEY_ESCAPE == aKey && GLFW_PRESS == aAction )
		{
			glfwSetWindowShouldClose( aWindow, GLFW_TRUE );
			return;
		}

		if (auto* state = static_cast<State_*>(glfwGetWindowUserPointer(aWindow)))
		{
			// camera controls if camera is active
			if (GLFW_KEY_W == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camControl.actionForward = true;
				else if (GLFW_RELEASE == aAction)
					state->camControl.actionForward = false;
			}
			else if (GLFW_KEY_S == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camControl.actionBackward = true;
				else if (GLFW_RELEASE == aAction)
					state->camControl.actionBackward = false;
			}
			else if (GLFW_KEY_A == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camControl.actionLeft = true;
				else if (GLFW_RELEASE == aAction)
					state->camControl.actionLeft = false;
			}
			else if (GLFW_KEY_D == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camControl.actionRight = true;
				else if (GLFW_RELEASE == aAction)
					state->camControl.actionRight = false;
			}
			else if (GLFW_KEY_Q == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camControl.actionDown = true;
				else if (GLFW_RELEASE == aAction)
					state->camControl.actionDown = false;
			}
			else if (GLFW_KEY_E == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camControl.actionUp = true;
				else if (GLFW_RELEASE == aAction)
					state->camControl.actionUp = false;
			}
			else if (GLFW_KEY_LEFT_SHIFT == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camControl.actionSpeedUp = true;
				else if (GLFW_RELEASE == aAction)
					state->camControl.actionSpeedUp = false;
			}
			else if (GLFW_KEY_LEFT_CONTROL == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camControl.actionSlowDown = true;
				else if (GLFW_RELEASE == aAction)
					state->camControl.actionSlowDown = false;
			}
		}
	}

	void glfw_callback_motion_(GLFWwindow* aWindow, double aX, double aY)
	{
		if (auto* state = static_cast<State_*>(glfwGetWindowUserPointer(aWindow)))
		{
			if (state->camControl.cameraActive)
			{
				auto const dx = float(aX - state->camControl.lastX);
				auto const dy = float(aY - state->camControl.lastY);

				state->camControl.phi += dx * kMouseSens;
				state->camControl.theta += dy * kMouseSens;
				
				if (state->camControl.theta > kPi/2.f)
					state->camControl.theta = kPi / 2.f;
				else if (state->camControl.theta < -kPi / 2.f)
					state->camControl.theta = -kPi / 2.f;
			}

			state->camControl.lastX = float(aX);
			state->camControl.lastY = float(aY);
		}
	}
}

namespace
{
	GLFWCleanupHelper::~GLFWCleanupHelper()
	{
		glfwTerminate();
	}

	GLFWWindowDeleter::~GLFWWindowDeleter()
	{
		if( window )
			glfwDestroyWindow( window );
	}
}
