#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <print>
#include <numbers>
#include <typeinfo>
#include <stdexcept>

#include <cstdlib>

// Imports
#include <algorithm>
#include <ctime> 
#include <vector> 
#include <cstdio>
#include <rapidobj/rapidobj.hpp> 
#include "../vmlib/vec2.hpp"
#include "../vmlib/vec3.hpp"
#include "../third_party/fontstash/include/fontstash.h"

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

	enum class CameraMode
	{
		Free = 0,
		Chase = 1,
		Ground = 2
	};

	struct Particle
	{
		Vec3f position;
		Vec3f velocity;
		float life;
		float maxLife;
	};

	struct ParticleSystem
	{
		std::vector<Particle> particles;
		GLuint vao = 0;
		GLuint texture = 0;
		float emissionTimer = 0.0f;

		ParticleSystem() { 
			particles.reserve(1024); 
		}
	};

	struct State_
	{
		ShaderProgram* progTex;
		ShaderProgram* progMat;

		struct UserInput {
			bool cameraActive;
			bool actionForward;
			bool actionBackward;
			bool actionLeft;
			bool actionRight;
			bool actionUp;
			bool actionDown;
			bool actionSpeedUp;
			bool actionSlowDown;
		} camInputs;

		struct CamCtrl_
		{
			float phi;
			float theta;

			float lastX;
			float lastY;

			Vec3f position;
		};
		CamCtrl_ camControl;
		CamCtrl_ camControlR;

		struct Animation_
		{
			bool isActive;
			bool isPlaying;
			float time;
			Vec3f startPosition;
		}animation;

		struct UI_
		{
			FONScontext* fs = nullptr;
			int font = FONS_INVALID;

			GLuint fontTexture = 0;
			int atlasW = 0;
			int atlasH = 0;

			GLuint program = 0;
			GLuint vao = 0;
			GLuint vbo = 0;
			GLint uScreen = -1;
			GLint uTex = -1;

			float mouseX = 0.f;
			float mouseY = 0.f;
			int winW = 0;
			int winH = 0;
		}ui;

		CameraMode cameraMode;
		ParticleSystem particles;
		CameraMode cameraModeR;
		bool splitScreen = false;
	};

	void glfw_callback_error_(int, char const*);

	void glfw_callback_mouse_button_(GLFWwindow*, int, int, int);
	void glfw_callback_key_(GLFWwindow*, int, int, int, int);
	void glfw_callback_motion_(GLFWwindow*, double, double);

	void ui_init(State_& state, int fbW, int fbH);
	void ui_cleanup(State_& state);
	void ui_resize(State_& state, int fbW, int fbH);
	void ui_draw(State_& state, int fbW, int fbH, float altitude);
	void ui_mouse_move(State_& state, float x, float y);
	void ui_mouse_button(State_& state, int button, int action);

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

	struct Material {
		Vec3f diffuse;
		float shine;
	};

	struct DirectionalLight {
		Vec3f direction;
		Vec3f color;
		bool enabled;
	} globalLight;

	struct PointLight {
		Vec3f position;
		Vec3f color;
		bool enabled;
	} pointLights[3];

	// Common info required to draw an object
	struct RenderContext {
		Mat44f projection;
		Mat44f cameraView;
		Vec3f camPos;
	};

	// Data for terrain and vehicle
	struct DefaultData {
		GLuint vao;
		std::size_t vertexCount;
		GLuint texture;
		Mat44f model;
	};
	// Data for pad
	struct PadData {
		GLuint vao;
		std::size_t vertexCount;
		std::vector<Material> materials;
	};

	SimpleMeshData load_wavefront_obj(char const* path, std::vector<Material>* materials = nullptr)
	{
		auto result = rapidobj::ParseFile(path);
		if (result.error)
			throw Error("Unable to load OBJ file '{}': {}", path, result.error.code.message());

		rapidobj::Triangulate(result);

		SimpleMeshData ret;

		// only if required
		if (materials)
		{
			materials->resize(result.materials.size());
			for (size_t i = 0; i < result.materials.size(); ++i)
			{
				auto const& mat = result.materials[i];
				(*materials)[i].diffuse = Vec3f{mat.diffuse[0], mat.diffuse[1], mat.diffuse[2]};
				(*materials)[i].shine = float(mat.shininess);
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
				if (materials)
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

	void setLighting(GLuint programId, DirectionalLight const& globalLight, PointLight const* pointLights)
	{
		// Global Directional Light
		GLint locDir = glGetUniformLocation(programId, "uGlobalLight.direction");
		GLint locColor = glGetUniformLocation(programId, "uGlobalLight.color");
		GLint locEnabled = glGetUniformLocation(programId, "uGlobalLight.enabled");

		glUniform3fv(locDir, 1, &globalLight.direction.x);
		glUniform3fv(locColor, 1, &globalLight.color.x);
		glUniform1i(locEnabled, globalLight.enabled);

		// 3 Local Point Lights
		std::string base;
		GLint locPos;
		for (std::size_t i = 0; i < 3; ++i)
		{
			base = "uPointLights[" + std::to_string(i) + "].";
			locPos = glGetUniformLocation(programId, (base + "position").c_str());
			locColor = glGetUniformLocation(programId, (base + "color").c_str());
			locEnabled = glGetUniformLocation(programId, (base + "enabled").c_str());

			glUniform3fv(locPos, 1, &pointLights[i].position.x);
			glUniform3fv(locColor, 1, &pointLights[i].color.x);
			glUniform1i(locEnabled, pointLights[i].enabled);
		}
	}

	void drawTerrain(
		RenderContext const& ctx,
		GLuint programId,
		GLuint texture,
		GLuint vao,
		std::size_t vertexCount
	)
	{
		Mat44f model = kIdentity44f;
		Mat44f mvp = ctx.projection * ctx.cameraView * model;
		Mat33f normalMatrix = mat44_to_mat33(transpose(invert(model)));

		glUseProgram(programId);
		setLighting(programId, globalLight, pointLights);
		glUniformMatrix4fv(0, 1, GL_TRUE, mvp.v);
		glUniformMatrix3fv(1, 1, GL_TRUE, normalMatrix.v);
		glUniformMatrix4fv(2, 1, GL_TRUE, model.v);
		glUniform3f(4, 0.05f, 0.05f, 0.05f);
		glUniform1i(5, true);
		glUniform3f(6, ctx.camPos.x, ctx.camPos.y, ctx.camPos.z);

		// Bind texture to texture unit0 and set sampler uniform
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glUniform1i(glGetUniformLocation(programId, "uTexture"), 0);

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, GLsizei(vertexCount));
		glBindVertexArray(0);
	}

	void drawLandingPad(
		RenderContext const& ctx,
		GLuint programId,
		Mat44f const& model,
		std::vector<Material> const& materials,
		GLuint vao,
		std::size_t vertexCount
	)
	{
		Mat44f mvp = ctx.projection * ctx.cameraView * model;
		Mat33f normalMatrix = mat44_to_mat33(transpose(invert(model)));

		glUseProgram(programId);
		setLighting(programId, globalLight, pointLights);
		glUniformMatrix4fv(0, 1, GL_TRUE, mvp.v);
		glUniformMatrix3fv(1, 1, GL_TRUE, normalMatrix.v);
		glUniform3f(4, 0.05f, 0.05f, 0.05f);
		glUniform3f(6, ctx.camPos.x, ctx.camPos.y, ctx.camPos.z);

		// Pass material colors
		GLint loc;
		std::string name;
		for (size_t i = 0; i < materials.size(); ++i)
		{
			name = "uMaterialDiffuse[" + std::to_string(i) + "]";
			loc = glGetUniformLocation(programId, name.c_str());
			glUniform3fv(loc, 1, &materials[i].diffuse.x);

			name = "uMaterialShine[" + std::to_string(i) + "]";
			loc = glGetUniformLocation(programId, name.c_str());
			glUniform1f(loc, materials[i].shine);
		}

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, GLsizei(vertexCount));
		glBindVertexArray(0);
	}

	void drawSpaceVehicle(
		RenderContext const& ctx,
		GLuint programId,
		Mat44f const& model,
		GLuint vao,
		std::size_t vertexCount
	)
	{
		Mat44f mvp = ctx.projection * ctx.cameraView * model;
		Mat33f normalMatrix = mat44_to_mat33(transpose(invert(model)));

		glUseProgram(programId);
		glUniformMatrix4fv(0, 1, GL_TRUE, mvp.v);
		glUniformMatrix3fv(1, 1, GL_TRUE, normalMatrix.v);
		setLighting(programId, globalLight, pointLights);
		glUniform3f(4, 0.05f, 0.05f, 0.05f);
		glUniform1i(5, false);

		glDisable(GL_CULL_FACE);

		glBindVertexArray(vao);
		glDrawArrays(GL_TRIANGLES, 0, GLsizei(vertexCount));
		glBindVertexArray(0);
	}

	void drawScene(
		RenderContext const& ctx,
		DefaultData const& terrain,
		PadData const& pad,
		DefaultData const& vehicle,
		GLuint const& defaultProgId,
		GLuint const& padProgId
	)
	{
		Mat44f padModel;

		drawTerrain(ctx, defaultProgId, terrain.texture, terrain.vao, terrain.vertexCount);

		padModel = make_translation(Vec3f{ 10.f, -0.97f, 45.f });
		drawLandingPad(ctx, padProgId, padModel, pad.materials, pad.vao, pad.vertexCount);

		padModel = make_translation(Vec3f{ 20.f, -0.97f, -50.f });
		drawLandingPad(ctx, padProgId, padModel, pad.materials, pad.vao, pad.vertexCount);

		drawSpaceVehicle(ctx, defaultProgId, vehicle.model, vehicle.vao, vehicle.vertexCount);
	}

	SimpleMeshData create_cylinder(float radius = 0.5f, float height = 1.0f, int segments = 32)
	{
		SimpleMeshData mesh;

		// side
		for (int i = 0; i < segments; ++i)
		{
			float a0 = (float(i) / segments) * 2.0f * kPi;
			float a1 = (float(i + 1) / segments) * 2.0f * kPi;

			float x0 = radius * std::cos(a0);
			float z0 = radius * std::sin(a0);
			float x1 = radius * std::cos(a1);
			float z1 = radius * std::sin(a1);

			Vec3f v0{ x0, 0.0f, z0 };
			Vec3f v1{ x0, height, z0 };
			Vec3f v2{ x1, 0.0f, z1 };
			Vec3f v3{ x1, height, z1 };

			Vec3f n0 = normalize(Vec3f{ x0, 0.0f, z0 });
			Vec3f n1 = normalize(Vec3f{ x1, 0.0f, z1 });

			// tri 1
			mesh.positions.push_back(v0);
			mesh.positions.push_back(v1);
			mesh.positions.push_back(v2);
			mesh.normals.push_back(n0);
			mesh.normals.push_back(n0);
			mesh.normals.push_back(n1);

			// tri 2
			mesh.positions.push_back(v1);
			mesh.positions.push_back(v3);
			mesh.positions.push_back(v2);
			mesh.normals.push_back(n0);
			mesh.normals.push_back(n1);
			mesh.normals.push_back(n1);
		}

		// top cap, normal +Y
		for (int i = 0; i < segments; ++i)
		{
			float a0 = (float(i) / segments) * 2.0f * kPi;
			float a1 = (float(i + 1) / segments) * 2.0f * kPi;

			Vec3f c{ 0.0f, height, 0.0f };
			Vec3f v1{ radius * std::cos(a1), height, radius * std::sin(a1) };
			Vec3f v2{ radius * std::cos(a0), height, radius * std::sin(a0) };

			mesh.positions.push_back(c);
			mesh.positions.push_back(v1);
			mesh.positions.push_back(v2);
			mesh.normals.push_back(Vec3f{ 0.0f, 1.0f, 0.0f });
			mesh.normals.push_back(Vec3f{ 0.0f, 1.0f, 0.0f });
			mesh.normals.push_back(Vec3f{ 0.0f, 1.0f, 0.0f });
		}

		// bottom cap, normal -Y
		for (int i = 0; i < segments; ++i)
		{
			float a0 = (float(i) / segments) * 2.0f * kPi;
			float a1 = (float(i + 1) / segments) * 2.0f * kPi;

			Vec3f c{ 0.0f, 0.0f, 0.0f };
			Vec3f v1{ radius * std::cos(a0), 0.0f, radius * std::sin(a0) };
			Vec3f v2{ radius * std::cos(a1), 0.0f, radius * std::sin(a1) };

			mesh.positions.push_back(c);
			mesh.positions.push_back(v1);
			mesh.positions.push_back(v2);
			mesh.normals.push_back(Vec3f{ 0.0f, -1.0f, 0.0f });
			mesh.normals.push_back(Vec3f{ 0.0f, -1.0f, 0.0f });
			mesh.normals.push_back(Vec3f{ 0.0f, -1.0f, 0.0f });
		}

		return mesh;
	}

	SimpleMeshData create_box(float width = 1.0f, float height = 1.0f, float depth = 1.0f)
	{
		SimpleMeshData mesh;

		float w = width * 0.5f;
		float h = height * 0.5f;
		float d = depth * 0.5f;

		// Front face (+Z)
		mesh.positions.push_back(Vec3f{ -w, -h, d });
		mesh.positions.push_back(Vec3f{ w, -h, d });
		mesh.positions.push_back(Vec3f{ w, h, d });
		mesh.normals.push_back(Vec3f{ 0, 0, 1 });
		mesh.normals.push_back(Vec3f{ 0, 0, 1 });
		mesh.normals.push_back(Vec3f{ 0, 0, 1 });

		mesh.positions.push_back(Vec3f{ -w, -h, d });
		mesh.positions.push_back(Vec3f{ w, h, d });
		mesh.positions.push_back(Vec3f{ -w, h, d });
		mesh.normals.push_back(Vec3f{ 0, 0, 1 });
		mesh.normals.push_back(Vec3f{ 0, 0, 1 });
		mesh.normals.push_back(Vec3f{ 0, 0, 1 });

		// Back face (-Z)
		mesh.positions.push_back(Vec3f{ w, -h, -d });
		mesh.positions.push_back(Vec3f{ -w, -h, -d });
		mesh.positions.push_back(Vec3f{ -w, h, -d });
		mesh.normals.push_back(Vec3f{ 0, 0, -1 });
		mesh.normals.push_back(Vec3f{ 0, 0, -1 });
		mesh.normals.push_back(Vec3f{ 0, 0, -1 });

		mesh.positions.push_back(Vec3f{ w, -h, -d });
		mesh.positions.push_back(Vec3f{ -w, h, -d });
		mesh.positions.push_back(Vec3f{ w, h, -d });
		mesh.normals.push_back(Vec3f{ 0, 0, -1 });
		mesh.normals.push_back(Vec3f{ 0, 0, -1 });
		mesh.normals.push_back(Vec3f{ 0, 0, -1 });

		// Top face (+Y)
		mesh.positions.push_back(Vec3f{ -w, h, d });
		mesh.positions.push_back(Vec3f{ w, h, d });
		mesh.positions.push_back(Vec3f{ w, h, -d });
		mesh.normals.push_back(Vec3f{ 0, 1, 0 });
		mesh.normals.push_back(Vec3f{ 0, 1, 0 });
		mesh.normals.push_back(Vec3f{ 0, 1, 0 });

		mesh.positions.push_back(Vec3f{ -w, h, d });
		mesh.positions.push_back(Vec3f{ w, h, -d });
		mesh.positions.push_back(Vec3f{ -w, h, -d });
		mesh.normals.push_back(Vec3f{ 0, 1, 0 });
		mesh.normals.push_back(Vec3f{ 0, 1, 0 });
		mesh.normals.push_back(Vec3f{ 0, 1, 0 });

		// Bottom face (-Y)
		mesh.positions.push_back(Vec3f{ -w, -h, -d });
		mesh.positions.push_back(Vec3f{ w, -h, -d });
		mesh.positions.push_back(Vec3f{ w, -h, d });
		mesh.normals.push_back(Vec3f{ 0, -1, 0 });
		mesh.normals.push_back(Vec3f{ 0, -1, 0 });
		mesh.normals.push_back(Vec3f{ 0, -1, 0 });

		mesh.positions.push_back(Vec3f{ -w, -h, -d });
		mesh.positions.push_back(Vec3f{ w, -h, d });
		mesh.positions.push_back(Vec3f{ -w, -h, d });
		mesh.normals.push_back(Vec3f{ 0, -1, 0 });
		mesh.normals.push_back(Vec3f{ 0, -1, 0 });
		mesh.normals.push_back(Vec3f{ 0, -1, 0 });

		// Right face (+X)
		mesh.positions.push_back(Vec3f{ w, -h, d });
		mesh.positions.push_back(Vec3f{ w, -h, -d });
		mesh.positions.push_back(Vec3f{ w, h, -d });
		mesh.normals.push_back(Vec3f{ 1, 0, 0 });
		mesh.normals.push_back(Vec3f{ 1, 0, 0 });
		mesh.normals.push_back(Vec3f{ 1, 0, 0 });

		mesh.positions.push_back(Vec3f{ w, -h, d });
		mesh.positions.push_back(Vec3f{ w, h, -d });
		mesh.positions.push_back(Vec3f{ w, h, d });
		mesh.normals.push_back(Vec3f{ 1, 0, 0 });
		mesh.normals.push_back(Vec3f{ 1, 0, 0 });
		mesh.normals.push_back(Vec3f{ 1, 0, 0 });

		// Left face (-X)
		mesh.positions.push_back(Vec3f{ -w, -h, -d });
		mesh.positions.push_back(Vec3f{ -w, -h, d });
		mesh.positions.push_back(Vec3f{ -w, h, d });
		mesh.normals.push_back(Vec3f{ -1, 0, 0 });
		mesh.normals.push_back(Vec3f{ -1, 0, 0 });
		mesh.normals.push_back(Vec3f{ -1, 0, 0 });

		mesh.positions.push_back(Vec3f{ -w, -h, -d });
		mesh.positions.push_back(Vec3f{ -w, h, d });
		mesh.positions.push_back(Vec3f{ -w, h, -d });
		mesh.normals.push_back(Vec3f{ -1, 0, 0 });
		mesh.normals.push_back(Vec3f{ -1, 0, 0 });
		mesh.normals.push_back(Vec3f{ -1, 0, 0 });

		return mesh;
	}

	SimpleMeshData create_sphere(float radius = 0.5f, int segments = 32, int rings = 16)
	{
		SimpleMeshData mesh;

		for (int ring = 0; ring < rings; ++ring)
		{
			float phi0 = kPi * float(ring) / rings;
			float phi1 = kPi * float(ring + 1) / rings;

			for (int seg = 0; seg < segments; ++seg)
			{
				float theta0 = 2.0f * kPi * float(seg) / segments;
				float theta1 = 2.0f * kPi * float(seg + 1) / segments;

				Vec3f v0{
					radius * std::sin(phi0) * std::cos(theta0),
					radius * std::cos(phi0),
					radius * std::sin(phi0) * std::sin(theta0)
				};
				Vec3f v1{
					radius * std::sin(phi0) * std::cos(theta1),
					radius * std::cos(phi0),
					radius * std::sin(phi0) * std::sin(theta1)
				};
				Vec3f v2{
					radius * std::sin(phi1) * std::cos(theta1),
					radius * std::cos(phi1),
					radius * std::sin(phi1) * std::sin(theta1)
				};
				Vec3f v3{
					radius * std::sin(phi1) * std::cos(theta0),
					radius * std::cos(phi1),
					radius * std::sin(phi1) * std::sin(theta0)
				};

				mesh.positions.push_back(v0);
				mesh.positions.push_back(v1);
				mesh.positions.push_back(v2);
				mesh.normals.push_back(normalize(v0));
				mesh.normals.push_back(normalize(v1));
				mesh.normals.push_back(normalize(v2));

				mesh.positions.push_back(v0);
				mesh.positions.push_back(v2);
				mesh.positions.push_back(v3);
				mesh.normals.push_back(normalize(v0));
				mesh.normals.push_back(normalize(v2));
				mesh.normals.push_back(normalize(v3));
			}
		}

		return mesh;
	}


	SimpleMeshData create_cone(float radius = 0.5f, float height = 1.0f, int segments = 32)
	{
		SimpleMeshData mesh;
		Vec3f apex{ 0.0f, height, 0.0f };

		// side
		for (int i = 0; i < segments; ++i)
		{
			float a0 = (float(i) / segments) * 2.0f * kPi;
			float a1 = (float(i + 1) / segments) * 2.0f * kPi;

			Vec3f v0{ radius * std::cos(a0), 0.0f, radius * std::sin(a0) };
			Vec3f v2{ radius * std::cos(a1), 0.0f, radius * std::sin(a1) };

			// arrange vertices so that normal points outward (fix culling issue)
			Vec3f e1 = apex - v0;
			Vec3f e2 = v2 - v0;
			Vec3f n = normalize(cross(e1, e2));

			mesh.positions.push_back(v0);
			mesh.positions.push_back(apex);
			mesh.positions.push_back(v2);
			mesh.normals.push_back(n);
			mesh.normals.push_back(n);
			mesh.normals.push_back(n);
		}

		// bottom cap, normal -Y
		for (int i = 0; i < segments; ++i)
		{
			float a0 = (float(i) / segments) * 2.0f * kPi;
			float a1 = (float(i + 1) / segments) * 2.0f * kPi;

			Vec3f c{ 0.0f, 0.0f, 0.0f };
			Vec3f v1{ radius * std::cos(a0), 0.0f, radius * std::sin(a0) };
			Vec3f v2{ radius * std::cos(a1), 0.0f, radius * std::sin(a1) };

			mesh.positions.push_back(c);
			mesh.positions.push_back(v1);
			mesh.positions.push_back(v2);
			mesh.normals.push_back(Vec3f{ 0.0f, -1.0f, 0.0f });
			mesh.normals.push_back(Vec3f{ 0.0f, -1.0f, 0.0f });
			mesh.normals.push_back(Vec3f{ 0.0f, -1.0f, 0.0f });
		}

		return mesh;
	}

	void append_transformed_mesh(SimpleMeshData& dest, const SimpleMeshData& src, const Mat44f& transform)
	{
		Mat33f normalTransform = mat44_to_mat33(transpose(invert(transform)));

		for (std::size_t i = 0; i < src.positions.size(); ++i)
		{
			Vec4f pos = transform * Vec4f{ src.positions[i].x, src.positions[i].y, src.positions[i].z, 1.0f };
			dest.positions.push_back(Vec3f{ pos.x, pos.y, pos.z });

			if (i < src.normals.size())
			{
				Vec3f normal = normalTransform * src.normals[i];
				dest.normals.push_back(normalize(normal));
			}
		}
	}

	SimpleMeshData create_space_vehicle()
	{
		SimpleMeshData vehicle;

		// sizes
		float bodyRadius = 1.0f;
		float bodyHeight = 4.0f;
		float coneHeight = 2.0f;
		float exhaustRad = 0.6f;
		float exhaustH = 1.0f;

		// Main body
		SimpleMeshData body = create_cylinder(bodyRadius, bodyHeight, 32);
		append_transformed_mesh(vehicle, body, kIdentity44f);

		// Nose cone on top
		SimpleMeshData nose = create_cone(bodyRadius, coneHeight, 32);
		Mat44f noseXform = make_translation(Vec3f{ 0.0f, bodyHeight, 0.0f });
		append_transformed_mesh(vehicle, nose, noseXform);

		// Exhaust cylinder at bottom
		SimpleMeshData exhaust = create_cylinder(exhaustRad, exhaustH, 32);
		Mat44f exhaustXform = make_translation(Vec3f{ 0.0f, -exhaustH, 0.0f });
		append_transformed_mesh(vehicle, exhaust, exhaustXform);

		// Cockpit sphere inside body
		SimpleMeshData cockpit = create_sphere(0.6f, 24, 16);
		Mat44f cockpitXform = make_translation(Vec3f{ 0.0f, 2.3f, 0.6f });
		append_transformed_mesh(vehicle, cockpit, cockpitXform);

		// Three protruded box engines around the base 
		for (int i = 0; i < 3; ++i)
		{
			float angle = (float(i) / 3.0f) * 2.0f * kPi;

			float finThickness = 0.2f;
			float finHeight = 1.5f;
			float finLength = 1.0f;

			SimpleMeshData fin = create_box(finThickness, finHeight, finLength);

			Mat44f finTranslate = make_translation(
				Vec3f{ bodyRadius + finThickness * 0.5f, finHeight * 0.5f, 0.0f }
			);

			Mat44f finLocalRotate = make_rotation_y(kPi * 0.5f);

			Mat44f finRotateAroundRocket = make_rotation_y(angle);

			Mat44f finXform = finRotateAroundRocket * finTranslate * finLocalRotate;

			append_transformed_mesh(vehicle, fin, finXform);
		}

		return vehicle;
	}

	struct AnimationState
	{
		Vec3f position;
		Vec3f direction;
		float speed;
	};

	AnimationState compute_vehicle_animation(float t, Vec3f const& startPos)
	{
		AnimationState result{};

		// Normalize time
		float flightDuration = 12.0f;
		float u = t / flightDuration;

		if (u < 0.0f) u = 0.0f;
		if (u > 1.0f) u = 1.0f;

		// Smooth Step
		float s = u * u * (3.0f - 2.0f * u);

		// Define Key Positions
		Vec3f p0 = startPos; // Start (Pad 1)
		Vec3f p3 = Vec3f{ 20.0f, -0.97f, -50.0f }; // End (Pad 2)

		// Control points
		float arcHeight = 40.0f;
		Vec3f p1 = p0 + Vec3f{ 0.0f, arcHeight, 0.0f };
		Vec3f p2 = p3 + Vec3f{ 0.0f, arcHeight, 0.0f };

		// Bezier Curve Calculation

		auto lerp = [](Vec3f const& a, Vec3f const& b, float t) -> Vec3f
			{
				return a + t * (b - a);
			};

		// Layer 1
		Vec3f A = lerp(p0, p1, s);
		Vec3f B = lerp(p1, p2, s);
		Vec3f C = lerp(p2, p3, s);

		// Layer 2
		Vec3f D = lerp(A, B, s);
		Vec3f E = lerp(B, C, s);

		// Layer 3 
		result.position = lerp(D, E, s);

		// Direction
		Vec3f tangent = E - D;

		if (length(tangent) > 0.001f)
		{
			result.direction = normalize(tangent);
		}
		else
		{
			result.direction = Vec3f{ 0.0f, 1.0f, 0.0f };
		}

		result.speed = 30.0f * (s * (1.0f - s)) * 4.0f;

		return result;
	}

	GLuint create_procedural_texture()
	{
		constexpr int size = 64;
		std::vector<unsigned char> data(size * size * 4);

		for (int y = 0; y < size; ++y)
		{
			for (int x = 0; x < size; ++x)
			{
				float dx = (float)x / size - 0.5f;
				float dy = (float)y / size - 0.5f;
				float dist = std::sqrt(dx * dx + dy * dy);
				float intensity = 1.0f - (dist * 2.0f);
				if (intensity < 0.0f) intensity = 0.0f;
				intensity = intensity * intensity;
				unsigned char val = static_cast<unsigned char>(intensity * 255.0f);
				int idx = (y * size + x) * 4;
				data[idx + 0] = val;
				data[idx + 1] = val;
				data[idx + 2] = val;
				data[idx + 3] = val;
			}
		}

		GLuint tex = 0;
		glGenTextures(1, &tex);
		glBindTexture(GL_TEXTURE_2D, tex);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8, size, size, 0, GL_RGBA, GL_UNSIGNED_BYTE, data.data());
		glGenerateMipmap(GL_TEXTURE_2D);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

		return tex;
	}

	GLuint create_particle_quad_vao()
	{
		float vertices[] = {
			-0.5f, -0.5f, 0.0f,   0.f, 1.f, 0.f,   0.0f, 0.0f,
			 0.5f, -0.5f, 0.0f,   0.f, 1.f, 0.f,   1.0f, 0.0f,
			 0.5f,  0.5f, 0.0f,   0.f, 1.f, 0.f,   1.0f, 1.0f,
			-0.5f,  0.5f, 0.0f,   0.f, 1.f, 0.f,   0.0f, 1.0f
		};
		unsigned int indices[] = { 0, 1, 2, 2, 3, 0 };

		GLuint vao = 0, vbo = 0, ebo = 0;
		glGenVertexArrays(1, &vao);
		glGenBuffers(1, &vbo);
		glGenBuffers(1, &ebo);

		glBindVertexArray(vao);

		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);

		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(indices), indices, GL_STATIC_DRAW);

		// Loc 0: position
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(0);
		// Loc 1: normal
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(3 * sizeof(float)));
		glEnableVertexAttribArray(1);
		// Loc 2: uv
		glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(float), (void*)(6 * sizeof(float)));
		glEnableVertexAttribArray(2);

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
		return vao;
	}

	void update_particles(State_& state, float dt, Vec3f const& vehiclePos,
		Mat44f const& vehicleTransform, bool emit)
	{
		auto& ps = state.particles;

		if (emit)
		{
			Vec3f exhaustOffset{ 0.f, -2.0f, 0.f };
			Vec4f exhaustPos4 = vehicleTransform * Vec4f{ exhaustOffset.x, exhaustOffset.y, exhaustOffset.z, 1.0f };
			Vec3f exhaustPos{ exhaustPos4.x, exhaustPos4.y, exhaustPos4.z };

			const float emissionRate = 100.0f; // particles per second
			int particlesToEmit = static_cast<int>(dt * emissionRate);

			for (int i = 0; i < particlesToEmit; ++i)
			{
				// Cap particle count
				if (ps.particles.size() >= 1024) break;

				Particle p;
				p.position = exhaustPos;

				// Random spread and speed variation of particles
				float spreadX = ((rand() % 200) - 100) / 100.0f;
				float spreadZ = ((rand() % 200) - 100) / 100.0f;
				float speedVariation = ((rand() % 100) / 100.0f); 

				p.velocity = Vec3f{
					spreadX * 1.5f,
					-3.0f - speedVariation * 2.0f,
					spreadZ * 1.5f
				};

				// Randomise TTL
				p.life = p.maxLife = 1.0f + ((rand() % 100) / 100.0f) * 0.5f; 
				ps.particles.push_back(p);
			}
		}
		else if (!state.animation.isActive)
		{
			ps.particles.clear();
			ps.emissionTimer = 0.0f;
		}

		// Particle physics 
		Vec3f gravity{ 0.0f, -1.0f, 0.0f }; 

		for (size_t i = 0; i < ps.particles.size(); )
		{
			auto& p = ps.particles[i];

			p.velocity += gravity * dt;
			p.position += p.velocity * dt;
			p.life -= dt;

			if (p.life <= 0.0f)
			{
				ps.particles[i] = ps.particles.back();
				ps.particles.pop_back();
			}
			else ++i;
		}
	}

	void draw_particles(State_& state, Mat44f const& view, Mat44f const& proj,
		Vec3f const& camRight, Vec3f const& camUp)
	{
		auto& ps = state.particles;
		if (ps.particles.empty()) return;

		glUseProgram(state.progTex->programId());

		glEnable(GL_BLEND);
		glBlendFunc(GL_ONE, GL_ONE); 
		glDepthMask(GL_FALSE); 
		glEnable(GL_DEPTH_TEST); 

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, ps.texture);
		glUniform1i(glGetUniformLocation(state.progTex->programId(), "uTexture"), 0);
		glUniform1i(5, true); 

		GLint origGlobalEnabled;
		glGetUniformiv(state.progTex->programId(),
			glGetUniformLocation(state.progTex->programId(), "uGlobalLight.enabled"),
			&origGlobalEnabled);

		// Disable lighting for particles
		glUniform3f(glGetUniformLocation(state.progTex->programId(), "uGlobalLight.color"), 0.f, 0.f, 0.f);
		glUniform1i(glGetUniformLocation(state.progTex->programId(), "uGlobalLight.enabled"), 0);

		for (int i = 0; i < 3; ++i)
		{
			std::string base = "uPointLights[" + std::to_string(i) + "].";
			glUniform1i(glGetUniformLocation(state.progTex->programId(), (base + "enabled").c_str()), 0);
		}

		glBindVertexArray(ps.vao);

		Vec3f camForward = normalize(cross(camUp, camRight));

		// Draw all particles
		for (const auto& p : ps.particles)
		{
			// Particle size variation
			float lifeRatio = p.life / p.maxLife;
			float scale = 0.5f * (0.3f + 0.7f * lifeRatio); // Scale from 0.3 to 1.0

			Mat44f model = kIdentity44f;

			model.v[0] = camRight.x * scale;
			model.v[4] = camRight.y * scale;
			model.v[8] = camRight.z * scale;
			model.v[12] = 0.0f;

			model.v[1] = camUp.x * scale;
			model.v[5] = camUp.y * scale;
			model.v[9] = camUp.z * scale;
			model.v[13] = 0.0f;

			model.v[2] = camForward.x * scale;
			model.v[6] = camForward.y * scale;
			model.v[10] = camForward.z * scale;
			model.v[14] = 0.0f;

			model.v[3] = p.position.x;
			model.v[7] = p.position.y;
			model.v[11] = p.position.z;
			model.v[15] = 1.0f;

			Mat44f mvp = proj * view * model;
			glUniformMatrix4fv(0, 1, GL_TRUE, mvp.v);
			glUniformMatrix4fv(2, 1, GL_TRUE, model.v);

			Mat33f normalMatrix = kIdentity33f;
			glUniformMatrix3fv(1, 1, GL_TRUE, normalMatrix.v);

			// Color Shift
			float intensity = lifeRatio * lifeRatio; 
			glUniform3f(4, 1.0f * intensity, 0.5f * intensity, 0.1f * intensity);

			glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);
		}

		glBindVertexArray(0);

		glDepthMask(GL_TRUE);
		glDisable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
	}

	struct CamFinal {
		Vec3f camPosFinal;
		Vec3f camForwardFinal;
		Vec3f camUpFinal;
		Vec3f camRightFinal;
	};

	CamFinal processCameraMode(
		CameraMode const& mode,
		Vec3f const& pos,
		Vec3f const& forward,
		Vec3f const& up,
		Vec3f const& right,
		State_::Animation_ const& animation,
		Vec3f const& currentVehiclePos
	)
	{
		CamFinal result;

		switch (mode)
		{

			case CameraMode::Free:
			{
				// Standard WASD controls
				result.camPosFinal = pos;
				result.camForwardFinal = forward;
				result.camUpFinal = up;
				result.camRightFinal = right;
				break;
			}

			case CameraMode::Chase:
			{
				// If animation is not active disable chase mode
				if (!animation.isActive)
				{
					result.camPosFinal = pos;
					result.camForwardFinal = forward;
					result.camUpFinal = up;
					result.camRightFinal = right;
					break;
				}

				// Normal chase mode when animation is active
				Vec3f target = currentVehiclePos;
				Vec3f vehicleDir = Vec3f{ 0.f, 1.f, 0.f };

				AnimationState as = compute_vehicle_animation(animation.time,
					animation.startPosition);
				if (length(as.direction) > 0.001f)
					vehicleDir = normalize(as.direction);

				result.camPosFinal = target - (vehicleDir * 20.0f) + Vec3f{ 0.0f, 5.0f, 0.0f };

				result.camForwardFinal = normalize(target - result.camPosFinal);
				result.camRightFinal = normalize(cross(result.camForwardFinal, Vec3f{ 0.f, 1.f, 0.f }));
				result.camUpFinal = normalize(cross(result.camRightFinal, result.camForwardFinal));
				break;
			}

			case CameraMode::Ground:
			{
				// FFixed spot on terrain looking at rocket
				result.camPosFinal = Vec3f{ 10.0f, 2.0f, 70.0f };

				Vec3f target = currentVehiclePos;

				result.camForwardFinal = normalize(target - result.camPosFinal);
				result.camRightFinal = normalize(cross(result.camForwardFinal, Vec3f{ 0.f, 1.f, 0.f }));
				result.camUpFinal = normalize(cross(result.camRightFinal, result.camForwardFinal));
				break;
			}
		}

		return result;
	}

	struct CamBasis {
		Vec3f forward;
		Vec3f right;
		Vec3f up;
	};

	CamBasis computeBasis(float phi, float theta)
	{
		CamBasis base;

		base.forward = {
			std::cos(theta) * std::sin(phi),
			std::sin(theta),
			std::cos(theta) * std::cos(phi)
		};
		base.forward = normalize(base.forward);

		base.right = cross(base.forward, Vec3f{ 0.f, 1.f, 0.f });
		base.right = normalize(base.right);

		base.up = cross(base.right, base.forward);
		base.up = normalize(base.up);

		return base;
	}

	struct UIVertex
	{
		float x, y;
		float u, v;
		unsigned char r, g, b, a;
	};

	std::vector<UIVertex> g_uiVerts;

	// Shader compilation and program linking helpers
	GLuint compile_shader_(GLenum type, const char* src)
	{
		GLuint sh = glCreateShader(type);
		glShaderSource(sh, 1, &src, nullptr);
		glCompileShader(sh);

		GLint ok = GL_FALSE;
		glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
		if (!ok)
		{
			char log[4096];
			GLsizei len = 0;
			glGetShaderInfoLog(sh, 4096, &len, log);
			throw Error("UI shader compile failed: {}", log);
		}
		return sh;
	}

	GLuint link_program_(GLuint vs, GLuint fs)
	{
		GLuint prog = glCreateProgram();
		glAttachShader(prog, vs);
		glAttachShader(prog, fs);
		glLinkProgram(prog);

		GLint ok = GL_FALSE;
		glGetProgramiv(prog, GL_LINK_STATUS, &ok);
		if (!ok)
		{
			char log[4096];
			GLsizei len = 0;
			glGetProgramInfoLog(prog, 4096, &len, log);
			throw Error("UI program link failed: {}", log);
		}
		return prog;
	}

	// Fontstash rendering callbacks
	// create font texture
	int fs_create(void* userPtr, int w, int h)
	{
		auto* state = static_cast<State_*>(userPtr);
		state->ui.atlasW = w;
		state->ui.atlasH = h;

		glGenTextures(1, &state->ui.fontTexture);
		glBindTexture(GL_TEXTURE_2D, state->ui.fontTexture);

		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

		glBindTexture(GL_TEXTURE_2D, 0);
		return 1;
	}

	// resize font texture
	int fs_resize(void* userPtr, int w, int h)
	{
		auto* state = static_cast<State_*>(userPtr);
		state->ui.atlasW = w;
		state->ui.atlasH = h;

		glBindTexture(GL_TEXTURE_2D, state->ui.fontTexture);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_R8, w, h, 0, GL_RED, GL_UNSIGNED_BYTE, nullptr);
		glBindTexture(GL_TEXTURE_2D, 0);
		return 1;
	}

	// update font texture region
	void fs_update(void* userPtr, int* rect, const unsigned char* data)
	{
		auto* state = static_cast<State_*>(userPtr);

		int x = rect[0];
		int y = rect[1];
		int w = rect[2] - rect[0];
		int h = rect[3] - rect[1];

		glBindTexture(GL_TEXTURE_2D, state->ui.fontTexture);
		glPixelStorei(GL_UNPACK_ALIGNMENT, 1);

		for (int row = 0; row < h; ++row)
		{
			const unsigned char* src = data + (y + row) * state->ui.atlasW + x;
			glTexSubImage2D(GL_TEXTURE_2D, 0, x, y + row, w, 1, GL_RED, GL_UNSIGNED_BYTE, src);
		}
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	// collect vertices for drawing
	void fs_draw(void*, const float* verts, const float* tcoords, const unsigned int* colors, int nverts)
	{
		for (int i = 0; i < nverts; ++i)
		{
			UIVertex v;
			v.x = verts[i * 2 + 0];
			v.y = verts[i * 2 + 1];
			v.u = tcoords[i * 2 + 0];
			v.v = tcoords[i * 2 + 1];

			unsigned int c = colors[i];
			v.a = (c >> 24) & 0xFF;
			v.b = (c >> 16) & 0xFF;
			v.g = (c >> 8) & 0xFF;
			v.r = (c >> 0) & 0xFF;

			g_uiVerts.push_back(v);
		}
	}

	// delete font texture
	void fs_delete(void* userPtr)
	{
		auto* state = static_cast<State_*>(userPtr);
		if (state->ui.fontTexture)
		{
			glDeleteTextures(1, &state->ui.fontTexture);
			state->ui.fontTexture = 0;
		}
	}

	// UI initialization function
	void ui_init(State_& state, int fbW, int fbH)
	{
		state.ui.winW = fbW;
		state.ui.winH = fbH;

		FONSparams params{};
		params.width = 512;
		params.height = 512;
		params.flags = FONS_ZERO_TOPLEFT;
		params.userPtr = &state;
		params.renderCreate = fs_create;
		params.renderResize = fs_resize;
		params.renderUpdate = fs_update;
		params.renderDraw = fs_draw;
		params.renderDelete = fs_delete;

		state.ui.fs = fonsCreateInternal(&params);
		if (!state.ui.fs)
			throw Error("Failed to create Fontstash context");

		state.ui.font = fonsAddFont(state.ui.fs, "sans", "assets/cw2/DroidSansMonoDotted.ttf");
		if (state.ui.font == FONS_INVALID)
			throw Error("Failed to load UI font: assets/cw2/DroidSansMonoDotted.ttf");

		// UI shader
		const char* vsSrc = R"(
		#version 430
		layout(location=0) in vec2 aPos;
		layout(location=1) in vec2 aUV;
		layout(location=2) in vec4 aColor;

		uniform vec2 uScreen;
		out vec2 vUV;
		out vec4 vColor;

		void main() {
			// Convert screen coords (0..W, 0..H) to NDC (-1..1, 1..-1)
			vec2 ndc = vec2(aPos.x / uScreen.x * 2.0 - 1.0, 
						    1.0 - aPos.y / uScreen.y * 2.0);
			gl_Position = vec4(ndc, 0.0, 1.0);
			vUV = aUV;
			vColor = aColor;
		}
	)";

		// Fragment shader
		const char* fsSrc = R"(
		#version 430
		in vec2 vUV;
		in vec4 vColor;
		uniform sampler2D uTex;
		out vec4 FragColor;

		void main() {
			float cov = texture(uTex, vUV).r;
			FragColor = vec4(vColor.rgb, vColor.a * cov);
		}
	)";

		// Compile and link UI shader program
		GLuint vsh = compile_shader_(GL_VERTEX_SHADER, vsSrc);
		GLuint fsh = compile_shader_(GL_FRAGMENT_SHADER, fsSrc);
		state.ui.program = link_program_(vsh, fsh);
		glDeleteShader(vsh);
		glDeleteShader(fsh);

		state.ui.uScreen = glGetUniformLocation(state.ui.program, "uScreen");
		state.ui.uTex = glGetUniformLocation(state.ui.program, "uTex");

		glGenVertexArrays(1, &state.ui.vao);
		glGenBuffers(1, &state.ui.vbo);

		glBindVertexArray(state.ui.vao);
		glBindBuffer(GL_ARRAY_BUFFER, state.ui.vbo);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(UIVertex), (void*)0);

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(UIVertex), (void*)(2 * sizeof(float)));

		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 4, GL_UNSIGNED_BYTE, GL_TRUE, sizeof(UIVertex), (void*)(4 * sizeof(float)));

		glBindVertexArray(0);
		glBindBuffer(GL_ARRAY_BUFFER, 0);
	}

	// UI Cleanup
	void ui_cleanup(State_& state)
	{
		if (state.ui.fs)
		{
			fonsDeleteInternal(state.ui.fs);
			state.ui.fs = nullptr;
		}

		if (state.ui.vbo) glDeleteBuffers(1, &state.ui.vbo);
		if (state.ui.vao) glDeleteVertexArrays(1, &state.ui.vao);
		if (state.ui.program) glDeleteProgram(state.ui.program);

		state.ui.vbo = 0;
		state.ui.vao = 0;
		state.ui.program = 0;
	}

	// UI Event Handlers
	// Window resize
	void ui_resize(State_& state, int fbW, int fbH)
	{
		state.ui.winW = fbW;
		state.ui.winH = fbH;
	}

	// Mouse movement
	void ui_mouse_move(State_& state, float x, float y)
	{
		state.ui.mouseX = x;
		state.ui.mouseY = y;
	}

	// Rectangle Hitbox
	bool point_in_rect(float px, float py, float x, float y, float w, float h)
	{
		return px >= x && px <= x + w && py >= y && py <= y + h;
	}

	// Mouse button
	void ui_mouse_button(State_& state, int button, int action)
	{
		if (button != GLFW_MOUSE_BUTTON_LEFT || action != GLFW_PRESS)
			return;

		float bw = 100.f;
		float bh = 30.f;
		float y = float(state.ui.winH) - 60.f;

		float launchX = float(state.ui.winW) * 0.5f - 120.f;
		float resetX = float(state.ui.winW) * 0.5f + 20.f;

		// Launch Button
		if (point_in_rect(state.ui.mouseX, state.ui.mouseY, launchX, y, bw, bh))
		{
			if (!state.animation.isActive)
			{
				state.animation.isActive = true;
				state.animation.isPlaying = true;
				state.animation.time = 0.f;
			}
			else
			{
				state.animation.isPlaying = !state.animation.isPlaying;
			}
		}

		// Reset Button
		if (point_in_rect(state.ui.mouseX, state.ui.mouseY, resetX, y, bw, bh))
		{
			state.animation.isActive = false;
			state.animation.isPlaying = false;
			state.animation.time = 0.f;
		}
	}

	// Flush Vertex Buffer and draw UI
	void ui_flush_text(State_& state)
	{
		if (g_uiVerts.empty())
			return;

		glUseProgram(state.ui.program);

		glUniform2f(state.ui.uScreen, float(state.ui.winW), float(state.ui.winH));

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, state.ui.fontTexture);
		glUniform1i(state.ui.uTex, 0);

		glBindVertexArray(state.ui.vao);
		glBindBuffer(GL_ARRAY_BUFFER, state.ui.vbo);
		glBufferData(GL_ARRAY_BUFFER, g_uiVerts.size() * sizeof(UIVertex), g_uiVerts.data(), GL_STREAM_DRAW);

		glDrawArrays(GL_TRIANGLES, 0, GLsizei(g_uiVerts.size()));

		glBindVertexArray(0);
		glBindTexture(GL_TEXTURE_2D, 0);
		glUseProgram(0);
	}

	// Main UI drawing function
	void ui_draw(State_& state, int fbW, int fbH, float altitude)
	{
		state.ui.winW = fbW;
		state.ui.winH = fbH;

		GLboolean depthWas = glIsEnabled(GL_DEPTH_TEST);
		GLboolean blendWas = glIsEnabled(GL_BLEND);

		glDisable(GL_DEPTH_TEST);
		glEnable(GL_BLEND);
		glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

		g_uiVerts.clear();

		fonsClearState(state.ui.fs);
		fonsSetFont(state.ui.fs, state.ui.font);
		fonsSetColor(state.ui.fs, 0xFFFFFFFF);

		char buf[64];
		std::snprintf(buf, sizeof(buf), "ALTITUDE: %.1f", altitude);
		fonsSetSize(state.ui.fs, 20.f);
		fonsSetAlign(state.ui.fs, FONS_ALIGN_LEFT | FONS_ALIGN_TOP);
		fonsDrawText(state.ui.fs, 20.f, 20.f, buf, nullptr);

		fonsSetSize(state.ui.fs, 18.f);
		fonsSetAlign(state.ui.fs, FONS_ALIGN_LEFT | FONS_ALIGN_TOP);

		float y = float(fbH) - 60.f;
		float launchX = float(fbW) * 0.5f - 120.f;
		float resetX = float(fbW) * 0.5f + 20.f;

		fonsDrawText(state.ui.fs, launchX, y, "LAUNCH", nullptr);
		fonsDrawText(state.ui.fs, resetX, y, "RESET", nullptr);

		ui_flush_text(state);

		if (!blendWas) glDisable(GL_BLEND);
		if (depthWas) glEnable(GL_DEPTH_TEST);
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

	//glfwWindowHint( GLFW_RESIZABLE, GLFW_FALSE )

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

	// Initial Camera mode
	state.cameraMode = CameraMode::Free;

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
	// glDisable(GL_CULL_FACE);
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
	ShaderProgram progDefault({
		{ GL_VERTEX_SHADER, "assets/cw2/default.vert" },
		{ GL_FRAGMENT_SHADER, "assets/cw2/default.frag" }
	});
	state.progTex = &progDefault;

	ShaderProgram progPads({
		{GL_VERTEX_SHADER, "assets/cw2/material.vert"},
		{GL_FRAGMENT_SHADER, "assets/cw2/material.frag"}
	});
	state.progMat = &progPads;

	// UI Initialization
	glfwGetFramebufferSize(window, &iwidth, &iheight);
	ui_init(state, iwidth, iheight);
	
	// Particles Initialization
	std::srand(static_cast<unsigned>(std::time(nullptr)));
	state.particles.vao = create_particle_quad_vao();
	state.particles.texture = create_procedural_texture();

	// Initialize camera
	state.camControl.phi = 0.0f;
	state.camControl.theta = -0.5f; 

	// Initialize light sources
	globalLight  = { Vec3f{0.1f, 1.f, -1.f}, Vec3f{ 0.9f, 0.9f, 0.6f }, true };
	pointLights[0] = { Vec3f{10.f, 5.f, 50.f}, Vec3f{0.f, 1.f, 1.f}, true };
	pointLights[1] = { Vec3f{15.f, 5.f, 42.f}, Vec3f{1.f, 1.f, 0.2f}, true };
	pointLights[2] = { Vec3f{5.f, 5.f, 42.f}, Vec3f{1.f, 0.f, 1.f}, true };

	// Animation state
	Vec3f vehiclePosition{ 10.f, -0.5f, 45.f };

	state.animation.isActive = false;
	state.animation.isPlaying = false;
	state.animation.time = 0.0f;
	state.animation.startPosition = vehiclePosition;

	auto last = Clock::now();

	float angle = 0.f;

	OGL_CHECKPOINT_ALWAYS();
	
	// Load terrain mesh and create VAO
	SimpleMeshData terrainMesh = load_wavefront_obj("assets/cw2/parlahti.obj");
	std::print("Loaded terrain mesh: {} vertices, {} texcoords\n", terrainMesh.positions.size(), terrainMesh.texcoords.size());
	GLuint terrainVAO = create_vao(terrainMesh);
	std::size_t terrainVertexCount = terrainMesh.positions.size();

	// Load landing_pad mesh and create VAO
	std::vector<Material> padMaterials;
	SimpleMeshData padMesh = load_wavefront_obj("assets/cw2/landingpad.obj", &padMaterials);
	std::print("Loaded landing_pad mesh: {} vertices, {} texcoords\n", padMesh.positions.size(), padMesh.texcoords.size());
	GLuint padVAO = create_vao(padMesh);
	std::size_t padVertexCount = padMesh.positions.size();

	// Create space vehicle mesh and create VAO
	SimpleMeshData vehicleMesh = create_space_vehicle();
	std::print("Created space vehicle: {} vertices\n", vehicleMesh.positions.size());
	GLuint vehicleVAO = create_vao(vehicleMesh);
	std::size_t vehicleVertexCount = vehicleMesh.positions.size();

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
			ui_resize(state, nwidth, nheight);
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
		auto& camR = state.camControlR;
		auto& input = state.camInputs;

		CamBasis basis = computeBasis(cam.phi, cam.theta);
		CamBasis basisR = computeBasis(camR.phi, cam.theta);

		float speed = kMovementSpeed;
		if (input.actionSpeedUp)
			speed *= 2.f;
		if (input.actionSlowDown)
			speed *= 0.5f;

		// Decide which screen can move
		bool moveLeft = (state.cameraMode == CameraMode::Free);
		bool rightFree = (state.cameraModeR == CameraMode::Free);
		bool moveRight = rightFree && state.splitScreen;

		float dtSpeed = speed * dt;
		if (input.actionForward)
		{
			if (moveLeft) cam.position += dtSpeed * basis.forward;
			if (moveRight) camR.position += dtSpeed * basisR.forward;
		}
		if (input.actionBackward)
		{
			if (moveLeft) cam.position -= dtSpeed * basis.forward;
			if (moveRight) camR.position -= dtSpeed * basisR.forward;
		}
		if (input.actionRight)
		{
			if (moveLeft) cam.position += dtSpeed * basis.right;
			if (moveRight) camR.position += dtSpeed * basisR.right;
		}
		if (input.actionLeft)
		{
			if (moveLeft) cam.position -= dtSpeed * basis.right;
			if (moveRight) camR.position -= dtSpeed * basisR.right;
		}
		if (input.actionUp)
		{
			if (moveLeft) cam.position += dtSpeed * basis.up;
			if (moveRight) camR.position += dtSpeed * basisR.up;
		}
			
		if (input.actionDown)
		{
			if (moveLeft) cam.position -= dtSpeed * basis.up;
			if (moveRight) camR.position -= dtSpeed * basisR.up;
		}
			

		auto& anim = state.animation;
		if (anim.isActive && anim.isPlaying)
		{
			anim.time += dt;
		}

		Vec3f currentVehiclePos;
		Mat44f vehicleModel;

		if (anim.isActive)
		{
			AnimationState animState = compute_vehicle_animation(anim.time, anim.startPosition);
			currentVehiclePos = animState.position;

			Vec3f dir = normalize(animState.direction);
			Vec3f worldUp{ 0.0f, 1.0f, 0.0f };

			if (std::abs(dot(dir, worldUp)) > 0.99f)
			{
				worldUp = Vec3f{ 1.0f, 0.0f, 0.0f };
			}

			Vec3f yAxis = dir;
			Vec3f zAxis = normalize(cross(yAxis, worldUp));
			Vec3f xAxis = normalize(cross(zAxis, yAxis));

			Mat44f rotation{
				xAxis.x, yAxis.x, zAxis.x, 0.0f,
				xAxis.y, yAxis.y, zAxis.y, 0.0f,
				xAxis.z, yAxis.z, zAxis.z, 0.0f,
				0.0f,    0.0f,    0.0f,    1.0f
			};

			vehicleModel = make_translation(currentVehiclePos)
				* rotation
				* make_scaling(0.5f, 0.5f, 0.5f);
		}
		else
		{
			currentVehiclePos = vehiclePosition;
			vehicleModel = make_translation(vehiclePosition)
				* make_scaling(0.5f, 0.5f, 0.5f)
				* make_rotation_y(kPi);
		}


		// Draw scene(s)
		OGL_CHECKPOINT_DEBUG();
		DefaultData terrain = { terrainVAO, terrainVertexCount, texture, kIdentity44f };
		PadData pad = { padVAO, padVertexCount, padMaterials };
		DefaultData vehicle = { vehicleVAO, vehicleVertexCount, 0, vehicleModel };

		// Update particles
		update_particles(state, dt, currentVehiclePos, vehicleModel, anim.isActive&& anim.isPlaying);

		// Render main or left screen
		CamFinal result = processCameraMode(state.cameraMode, cam.position, basis.forward, basis.up, basis.right, state.animation, currentVehiclePos);
		Mat44f camera_view = construct_camera_view(result.camForwardFinal, result.camUpFinal, result.camRightFinal, result.camPosFinal);

		float aspectRatio = fbwidth / float(fbheight);
		if (state.splitScreen)
			aspectRatio = (fbwidth * 0.5f) / fbheight;

		Mat44f projection = make_perspective_projection(
			60.f * kPi / 180.f,
			aspectRatio,
			0.1f, 1000.0f
		);

		// Clear and draw frame
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		float width = fbwidth;
		if (state.splitScreen)
			width = fbwidth * 0.5f;

		glViewport(0, 0, width, fbheight);
		RenderContext baseContext = { projection, camera_view, cam.position };
		drawScene(baseContext, terrain, pad, vehicle, progDefault.programId(), progPads.programId());
		draw_particles(state, camera_view, projection, result.camRightFinal, result.camUpFinal);

		// Render right screen if necessary
		if (state.splitScreen)
		{
			CamFinal resultR = processCameraMode(state.cameraModeR, camR.position, basisR.forward, basisR.up, basisR.right, state.animation, currentVehiclePos);
			Mat44f right_view = construct_camera_view(resultR.camForwardFinal, resultR.camUpFinal, resultR.camRightFinal, resultR.camPosFinal);

			float halfWidth = (fbwidth * 0.5f);

			float aspectRatio = halfWidth / fbheight;
			Mat44f projectionR = make_perspective_projection(
				60.f * kPi / 180.f,
				aspectRatio,
				0.1f, 1000.0f
			);

			glViewport(halfWidth, 0, halfWidth, fbheight);
			RenderContext baseContextR = { projectionR, right_view, camR.position };
			drawScene(baseContextR, terrain, pad, vehicle, progDefault.programId(), progPads.programId());
			draw_particles(state, right_view, projectionR, resultR.camRightFinal, resultR.camUpFinal);
		}

		// Draw UI overlay
		glViewport(0, 0, int(fbwidth), int(fbheight));
		ui_draw(state, int(fbwidth), int(fbheight), currentVehiclePos.y);

		glEnable(GL_CULL_FACE);

		glUseProgram(0);

		OGL_CHECKPOINT_DEBUG();

		// Display results
		glfwSwapBuffers( window );
	}

	// Cleanup.
	ui_cleanup(state);
	
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
	void updateCamRotation(double aX, double aY, State_::CamCtrl_& camera)
	{
		auto const dx = float(aX - camera.lastX);
		auto const dy = float(aY - camera.lastY);

		camera.phi -= dx * kMouseSens;
		camera.theta -= dy * kMouseSens;

		if (camera.theta > kPi / 2.f)
			camera.theta = kPi / 2.f;
		else if (camera.theta < -kPi / 2.f)
			camera.theta = -kPi / 2.f;
	}

	void updateCamMode(CameraMode& cameraMode)
	{
		int mode = static_cast<int>(cameraMode);
		mode = (mode + 1) % 3;
		cameraMode = static_cast<CameraMode>(mode);
	}

	void glfw_callback_error_( int aErrNum, char const* aErrDesc )
	{
		std::print( stderr, "GLFW error: {} ({})\n", aErrDesc, aErrNum );
	}

	void glfw_callback_mouse_button_(GLFWwindow* aWindow, int aButton, int aAction, int mod)
	{
		// Handle UI button clicks first
		if (auto* state = static_cast<State_*>(glfwGetWindowUserPointer(aWindow)))
		{
			ui_mouse_button(*state, aButton, aAction);
		}

		// activate / deactivate camera control
		if (GLFW_MOUSE_BUTTON_RIGHT == aButton && GLFW_PRESS == aAction)
		{
			auto* state = static_cast<State_*>(glfwGetWindowUserPointer(aWindow));
			if (state)
			{
				state->camInputs.cameraActive = !state->camInputs.cameraActive;

				if (state->camInputs.cameraActive)
					glfwSetInputMode(aWindow, GLFW_CURSOR, GLFW_CURSOR_HIDDEN);
				else
					glfwSetInputMode(aWindow, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
			}
		}
	}

	void glfw_callback_key_( GLFWwindow* aWindow, int aKey, int, int aAction, int mod)
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
					state->camInputs.actionForward = true;
				else if (GLFW_RELEASE == aAction)
					state->camInputs.actionForward = false;
			}
			else if (GLFW_KEY_S == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camInputs.actionBackward = true;
				else if (GLFW_RELEASE == aAction)
					state->camInputs.actionBackward = false;
			}
			else if (GLFW_KEY_A == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camInputs.actionLeft = true;
				else if (GLFW_RELEASE == aAction)
					state->camInputs.actionLeft = false;
			}
			else if (GLFW_KEY_D == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camInputs.actionRight = true;
				else if (GLFW_RELEASE == aAction)
					state->camInputs.actionRight = false;
			}
			else if (GLFW_KEY_Q == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camInputs.actionDown = true;
				else if (GLFW_RELEASE == aAction)
					state->camInputs.actionDown = false;
			}
			else if (GLFW_KEY_E == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camInputs.actionUp = true;
				else if (GLFW_RELEASE == aAction)
					state->camInputs.actionUp = false;
			}
			else if (GLFW_KEY_LEFT_SHIFT == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camInputs.actionSpeedUp = true;
				else if (GLFW_RELEASE == aAction)
					state->camInputs.actionSpeedUp = false;
			}
			else if (GLFW_KEY_LEFT_CONTROL == aKey)
			{
				if (GLFW_PRESS == aAction)
					state->camInputs.actionSlowDown = true;
				else if (GLFW_RELEASE == aAction)
					state->camInputs.actionSlowDown = false;
			}
			// animation controls
			if (GLFW_KEY_F == aKey && GLFW_PRESS == aAction)
			{
				if (!state->animation.isActive)
				{
					state->animation.isActive = true;
					state->animation.isPlaying = true;
					state->animation.time = 0.0f;
				}
				else
				{
					state->animation.isPlaying = !state->animation.isPlaying;
				}
			}
			else if (GLFW_KEY_R == aKey && GLFW_PRESS == aAction)
			{
				state->animation.isActive = false;
				state->animation.isPlaying = false;
				state->animation.time = 0.0f;
			}

			// Camera Control Toggle
			if (GLFW_KEY_C == aKey && GLFW_PRESS == aAction)
			{
				if ((GLFW_MOD_SHIFT & mod) && state->splitScreen)
					updateCamMode(state->cameraModeR);
				else
					updateCamMode(state->cameraMode);
			}

			// Split Screen Toggle
			if (GLFW_KEY_V == aKey && GLFW_PRESS == aAction)
				state->splitScreen = !state->splitScreen;
		}

		if (GLFW_PRESS == aAction)
		{
			if (GLFW_KEY_1 == aKey)
				pointLights[0].enabled = !pointLights[0].enabled;
			else if (GLFW_KEY_2 == aKey)
				pointLights[1].enabled = !pointLights[1].enabled;
			else if (GLFW_KEY_3 == aKey)
				pointLights[2].enabled = !pointLights[2].enabled;
			else if (GLFW_KEY_4 == aKey)
				globalLight.enabled = !globalLight.enabled;
		}
	}

	void glfw_callback_motion_(GLFWwindow* aWindow, double aX, double aY)
	{
		if (auto* state = static_cast<State_*>(glfwGetWindowUserPointer(aWindow)))
		{
			ui_mouse_move(*state, float(aX), float(aY));

			if (state->camInputs.cameraActive)
				updateCamRotation(aX, aY, state->camControl);
			state->camControl.lastX = float(aX);
			state->camControl.lastY = float(aY);

			if (state->splitScreen && state->camInputs.cameraActive)
				updateCamRotation(aX, aY, state->camControlR);
			state->camControlR.lastX = float(aX);
			state->camControlR.lastY = float(aY);
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
