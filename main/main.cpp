#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <print>
#include <numbers>
#include <typeinfo>
#include <stdexcept>

#include <cstdlib>

// Imports
#include <vector> 
#include <rapidobj/rapidobj.hpp> 
#include "../vmlib/vec2.hpp"
#include "../vmlib/vec3.hpp"

#include "../support/error.hpp"
#include "../support/program.hpp"
#include "../support/checkpoint.hpp"
#include "../support/debug_output.hpp"

#include "../vmlib/vec4.hpp"
#include "../vmlib/mat44.hpp"
#include "../vmlib/mat33.hpp"

#include "defaults.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "../third_party/stb/include/stb_image.h"


constexpr float kPi = std::numbers::pi_v<float>;

namespace
{
	constexpr char const* kWindowTitle = "COMP3811 - CW2";

	constexpr float kMovementSpeed = 5.f;
	constexpr float kMouseSens = 0.01f;

	struct State_
	{
		ShaderProgram* progTex;
		ShaderProgram* progMat;

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

	struct SimpleMeshData
	{
		std::vector<Vec3f> positions;
		std::vector<Vec3f> normals;
		std::vector<Vec2f> texcoords;
		std::vector<float> materialIds;
	};

	SimpleMeshData load_wavefront_obj(char const* path, std::vector<Vec3f>* materialColors = nullptr)
	{
		auto result = rapidobj::ParseFile(path);
		if (result.error)
			throw Error("Unable to load OBJ file '{}': {}", path, result.error.code.message());

		rapidobj::Triangulate(result);

		SimpleMeshData ret;

		// only if required
		if (materialColors)
		{
			materialColors->resize(result.materials.size());
			for (size_t i = 0; i < result.materials.size(); ++i)
			{
				auto const& mat = result.materials[i];
				(*materialColors)[i] = Vec3f{mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]};
			}
		}

		for (auto const& shape : result.shapes)
		{
			for (std::size_t i = 0; i < shape.mesh.indices.size(); ++i)
			{
				auto const& idx = shape.mesh.indices[i];

				// Positions
				ret.positions.emplace_back(Vec3f{
					result.attributes.positions[idx.position_index * 3 + 0],
					result.attributes.positions[idx.position_index * 3 + 1],
					result.attributes.positions[idx.position_index * 3 + 2]
					});

				// Normals
				if (idx.normal_index >= 0)
				{
					ret.normals.emplace_back(normalize(Vec3f{
						result.attributes.normals[idx.normal_index * 3 + 0],
						result.attributes.normals[idx.normal_index * 3 + 1],
						result.attributes.normals[idx.normal_index * 3 + 2]
						}));
				}

				// Texture coordinates
				if (idx.texcoord_index >= 0)
				{
					ret.texcoords.emplace_back(Vec2f{
						result.attributes.texcoords[idx.texcoord_index * 2 + 0],
						result.attributes.texcoords[idx.texcoord_index * 2 + 1]
						});
				}

				// Materials
				if (materialColors)
				{
					int faceIndex = int(i / 3);
					int matId = shape.mesh.material_ids[faceIndex];
					ret.materialIds.push_back(float(matId));
				}
			}
		}
		return ret;
	}

	GLuint create_vao(SimpleMeshData const& meshData)
	{
		GLuint vao = 0;
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		// Positions at location 0
		if (!meshData.positions.empty())
		{
			GLuint vbo = 0;
			glGenBuffers(1, &vbo);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(
				GL_ARRAY_BUFFER,
				meshData.positions.size() * sizeof(Vec3f),
				meshData.positions.data(),
				GL_STATIC_DRAW
			);
			glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
			glEnableVertexAttribArray(0);
		}

		// Normals at location 1
		if (!meshData.normals.empty())
		{
			GLuint vbo = 0;
			glGenBuffers(1, &vbo);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(
				GL_ARRAY_BUFFER,
				meshData.normals.size() * sizeof(Vec3f),
				meshData.normals.data(),
				GL_STATIC_DRAW
			);
			glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, nullptr);
			glEnableVertexAttribArray(1);
		}

		// Texcoords at location 2
		if (!meshData.texcoords.empty())
		{
			GLuint vbo = 0;
			glGenBuffers(1, &vbo);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(
				GL_ARRAY_BUFFER,
				meshData.texcoords.size() * sizeof(Vec2f),
				meshData.texcoords.data(),
				GL_STATIC_DRAW
			);
			glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 0, nullptr);
			glEnableVertexAttribArray(2);
		}

		// Material IDs at location 3
		if (!meshData.materialIds.empty())
		{
			GLuint vbo = 0;
			glGenBuffers(1, &vbo);
			glBindBuffer(GL_ARRAY_BUFFER, vbo);
			glBufferData(
				GL_ARRAY_BUFFER,
				meshData.materialIds.size() * sizeof(float),
				meshData.materialIds.data(),
				GL_STATIC_DRAW
			);
			glVertexAttribPointer(3, 1, GL_FLOAT, GL_FALSE, 0, nullptr);
			glEnableVertexAttribArray(3);
		}

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		return vao;
	}

	GLuint loadTexture(const char* filename)
	{
		int width, height, channels;
		stbi_uc* data = stbi_load(filename, &width, &height, &channels, 4);

		if (!data) {
			throw Error("Failed to load texture file '{}'", filename);
		}

		GLuint textureID;
		glGenTextures(1, &textureID);
		glBindTexture(GL_TEXTURE_2D, textureID);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_SRGB8_ALPHA8, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, data);
		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);

		stbi_image_free(data);

		return textureID;
	}


	void drawTerrain(
		Mat44f const& projection,
		Mat44f const& camera_view, 
		GLuint programId, 
		Vec3f const& lightDir,
		GLuint texture,
		GLuint vao,
		std::size_t vertexCount
	)
	{
		Mat44f model = kIdentity44f;
		Mat44f mvp = projection * camera_view * model;
		Mat33f normalMatrix = mat44_to_mat33(transpose(invert(model)));

		glUseProgram(programId);
		glUniformMatrix4fv(0, 1, GL_TRUE, mvp.v);
		glUniformMatrix3fv(1, 1, GL_TRUE, normalMatrix.v);
		glUniform3fv(2, 1, &lightDir.x);
		glUniform3f(3, 1.f, 1.f, 1.f);
		glUniform3f(4, 0.1f, 0.1f, 0.1f);

		// Bind texture to texture unit0 and set sampler uniform
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glUniform1i(glGetUniformLocation(programId, "uTexture"), 0);

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, GLsizei(vertexCount));
		glBindVertexArray(0);
	}

	void drawLandingPad(
		Mat44f const& projection,
		Mat44f const& camera_view,
		GLuint programId,
		Vec3f const& lightDir,
		Mat44f const& model,
		std::vector<Vec3f> const& colors,
		GLuint vao,
		std::size_t vertexCount
	)
	{
		Mat44f mvp = projection * camera_view * model;
		Mat33f normalMatrix = mat44_to_mat33(transpose(invert(model)));

		glUseProgram(programId);
		glUniformMatrix4fv(0, 1, GL_TRUE, mvp.v);
		glUniformMatrix3fv(1, 1, GL_TRUE, normalMatrix.v);
		glUniform3fv(2, 1, &lightDir.x);
		glUniform3f(3, 0.9f, 0.9f, 0.6f);
		glUniform3f(4, 0.05f, 0.05f, 0.05f);

		// Pass material colors
		for (size_t i = 0; i < colors.size(); ++i)
		{
			std::string name = "uMaterialDiffuse[" + std::to_string(i) + "]";
			GLint loc = glGetUniformLocation(programId, name.c_str());
			if (loc >= 0)
				glUniform3fv(loc, 1, &colors[i].x);
		}

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, GLsizei(vertexCount));
		glBindVertexArray(0);
	}
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

	// OPENGL State Setup
	glEnable(GL_FRAMEBUFFER_SRGB);   
	glEnable(GL_DEPTH_TEST);         
	glEnable(GL_CULL_FACE);          
	glClearColor(0.2f, 0.2f, 0.2f, 1.0f); 

	OGL_CHECKPOINT_ALWAYS();

	// Get actual framebuffer size.
	// This can be different from the window size, as standard window
	// decorations (title bar, borders, ...) may be included in the window size
	// but not be part of the drawable surface area.
	int iwidth, iheight;
	glfwGetFramebufferSize( window, &iwidth, &iheight );

	glViewport( 0, 0, iwidth, iheight );

	// Other initialization & loading
	
	// Load shader programs
	ShaderProgram progTerrain({
		{ GL_VERTEX_SHADER, "assets/cw2/default.vert" },
		{ GL_FRAGMENT_SHADER, "assets/cw2/default.frag" }
	});
	state.progTex = &progTerrain;

	ShaderProgram progPads({
		{GL_VERTEX_SHADER, "assets/cw2/material.vert"},
		{GL_FRAGMENT_SHADER, "assets/cw2/material.frag"}
	});
	state.progMat = &progPads;

	state.camControl.position = Vec3f{0.f,3.f,0.f }; 
	state.camControl.theta = -0.5f; 

	// Animation state
	auto last = Clock::now();

	float angle = 0.f;

	OGL_CHECKPOINT_ALWAYS();
	
	// Load terrain mesh and create VAO
	SimpleMeshData terrainMesh = load_wavefront_obj("assets/cw2/parlahti.obj");
	std::print("Loaded terrain mesh: {} vertices, {} texcoords\n", terrainMesh.positions.size(), terrainMesh.texcoords.size());
	GLuint terrainVAO = create_vao(terrainMesh);
	std::size_t terrainVertexCount = terrainMesh.positions.size();

	// Load landing_pad mesh and create VAO
	std::vector<Vec3f> padColors;
	SimpleMeshData padMesh = load_wavefront_obj("assets/cw2/landingpad.obj", &padColors);
	std::print("Loaded landing_pad mesh: {} vertices, {} texcoords\n", padMesh.positions.size(), padMesh.texcoords.size());
	GLuint padVAO = create_vao(padMesh);
	std::size_t padVertexCount = padMesh.positions.size();

	// Load texture
	GLuint texture = loadTexture("assets/cw2/L4343A-4k.jpeg");


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
		Vec3f lightDir = normalize(Vec3f{0.f,1.f, -1.f });
		
		// Clear and draw frame
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		Mat44f padModel = kIdentity44f;

		drawTerrain(projection, camera_view, progTerrain.programId(),
			lightDir, texture,
			terrainVAO, terrainVertexCount
		);
		padModel = make_translation(Vec3f{10.f, -0.95f, 45.f});
		drawLandingPad(projection, camera_view, progPads.programId(),
			lightDir, padModel, padColors,
			padVAO, padVertexCount
		);
		padModel = make_translation(Vec3f{20.f, -0.95f, -50.f});
		drawLandingPad(projection, camera_view, progPads.programId(),
			lightDir, padModel, padColors,
			padVAO, padVertexCount
		);
		glUseProgram(0);

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

				state->camControl.phi -= dx * kMouseSens;
				state->camControl.theta -= dy * kMouseSens;
				
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
