#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <print>
#include <numbers>
#include <typeinfo>
#include <stdexcept>

#include <cstdlib>

// Imports
#include <algorithm>
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

		struct Animation_
		{
			bool isActive;
			bool isPlaying;
			float time;
			Vec3f startPosition;
		}animation;
	};

	void glfw_callback_error_(int, char const*);

	void glfw_callback_mouse_button_(GLFWwindow*, int, int, int);
	void glfw_callback_key_(GLFWwindow*, int, int, int, int);
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
		Mat44f const& projection,
		Mat44f const& camera_view,
		GLuint programId,
		GLuint texture,
		GLuint vao,
		std::size_t vertexCount,
		DirectionalLight const& globalLight,
		PointLight const* pointLights,
		Vec3f const& cameraPos
	)
	{
		Mat44f model = kIdentity44f;
		Mat44f mvp = projection * camera_view * model;
		Mat33f normalMatrix = mat44_to_mat33(transpose(invert(model)));

		glUseProgram(programId);
		setLighting(programId, globalLight, pointLights);
		glUniformMatrix4fv(0, 1, GL_TRUE, mvp.v);
		glUniformMatrix3fv(1, 1, GL_TRUE, normalMatrix.v);
		glUniformMatrix4fv(2, 1, GL_TRUE, model.v);
		glUniform3f(4, 0.05f, 0.05f, 0.05f);
		glUniform1i(5, true);
		glUniform3f(6, cameraPos.x, cameraPos.y, cameraPos.z);

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
		Mat44f const& model,
		std::vector<Material> const& materials,
		GLuint vao,
		std::size_t vertexCount,
		DirectionalLight const& globalLight,
		PointLight const* pointLights,
		Vec3f const& cameraPos
	)
	{
		Mat44f mvp = projection * camera_view * model;
		Mat33f normalMatrix = mat44_to_mat33(transpose(invert(model)));

		glUseProgram(programId);
		setLighting(programId, globalLight, pointLights);
		glUniformMatrix4fv(0, 1, GL_TRUE, mvp.v);
		glUniformMatrix3fv(1, 1, GL_TRUE, normalMatrix.v);
		glUniform3f(4, 0.05f, 0.05f, 0.05f);
		glUniform3f(6, cameraPos.x, cameraPos.y, cameraPos.z);

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

	// Initialize camera
	state.camControl.position = Vec3f{0.f,3.f,0.f }; 
	state.camControl.theta = -0.5f; 

	// Initialize light sources
	float intensityMultiplier = 1;
	globalLight  = { Vec3f{0.1f, 1.f, -1.f}, Vec3f{ 0.9f, 0.9f, 0.6f }, true };
	pointLights[0] = { Vec3f{0.f, 30.f, 35.f}, intensityMultiplier * Vec3f{0.f, 1.f, 1.f}, true };
	pointLights[1] = { Vec3f{10.f, 30.f, 55.f}, intensityMultiplier * Vec3f{1.f, 1.f, 0.2f}, true };
	pointLights[2] = { Vec3f{20.f, 30.f, 35.f}, intensityMultiplier * Vec3f{1.f, 0.f, 1.f}, true };

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

		drawTerrain(projection, camera_view, progDefault.programId(),
			texture, terrainVAO, terrainVertexCount, globalLight, pointLights, cam.position
		);
		padModel = make_translation(Vec3f{10.f, -0.97f, 45.f});
		drawLandingPad(projection, camera_view, progPads.programId(),
			padModel, padMaterials,
			padVAO, padVertexCount, globalLight, pointLights, cam.position
		);
		padModel = make_translation(Vec3f{20.f, -0.97f, -50.f});
		drawLandingPad(projection, camera_view, progPads.programId(),
			padModel, padMaterials,
			padVAO, padVertexCount, globalLight, pointLights, cam.position
		);

		// Draw Space Vehicle

		// Mat44f vehicleModel = make_translation(vehiclePosition) * make_scaling(0.5f, 0.5f, 0.5f) * make_rotation_y(kPi);
		Mat44f vehicleMVP = projection * camera_view * vehicleModel;
		Mat33f vehicleNormalMatrix = mat44_to_mat33(transpose(invert(vehicleModel)));

		glUseProgram(progDefault.programId());
		glUniformMatrix4fv(0, 1, GL_TRUE, vehicleMVP.v);
		glUniformMatrix3fv(1, 1, GL_TRUE, vehicleNormalMatrix.v);
		setLighting(progDefault.programId(), globalLight, pointLights);
		glUniform3f(4, 0.05f, 0.05f, 0.05f);
		glUniform1i(5, false);

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, texture);
		glUniform1i(glGetUniformLocation(progDefault.programId(), "uTexture"), 0);

		glDisable(GL_CULL_FACE);

		glBindVertexArray(vehicleVAO);
		glDrawArrays(GL_TRIANGLES, 0, GLsizei(vehicleVertexCount));
		glBindVertexArray(0);

		glEnable(GL_CULL_FACE);

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
