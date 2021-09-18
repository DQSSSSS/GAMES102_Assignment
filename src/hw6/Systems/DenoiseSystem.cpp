#include "DenoiseSystem.h"

#include "../Components/DenoiseData.h"

#include <_deps/imgui/imgui.h>

#include <spdlog/spdlog.h>

using namespace Ubpa;

void meshToHE(std::shared_ptr<Ubpa::Utopia::Mesh> mesh, std::shared_ptr<HEMeshX> heMesh) {
	heMesh->Clear();
	if (!mesh) {
		spdlog::warn("mesh is nullptr");
		return;
	}

	if (mesh->GetSubMeshes().size() != 1) {
		spdlog::warn("number of submeshes isn't 1");
		return;
	}

//		data->copy = *data->mesh;

	std::vector<size_t> indices(mesh->GetIndices().begin(), mesh->GetIndices().end());

	//spdlog::info("size = {}", indices.size());

	heMesh->Init(indices, 3);
	if (!heMesh->IsTriMesh())
		spdlog::warn("HEMesh init fail");

	for (size_t i = 0; i < mesh->GetPositions().size(); i++)
		heMesh->Vertices().at(i)->position = mesh->GetPositions().at(i);

	//spdlog::info("Mesh to HEMesh success");
}

void heToMesh(std::shared_ptr<HEMeshX> heMesh, std::shared_ptr<Ubpa::Utopia::Mesh> mesh) {
	if (!mesh) {
		spdlog::warn("mesh is nullptr");
		return;
	}

	if (!heMesh->IsTriMesh() || heMesh->IsEmpty()) {
		spdlog::warn("HEMesh isn't triangle mesh or is empty");
		return;
	}

	mesh->SetToEditable();

	const size_t N = heMesh->Vertices().size();
	const size_t M = heMesh->Polygons().size();
	std::vector<Ubpa::pointf3> positions(N);
	std::vector<uint32_t> indices(M * 3);
	for (size_t i = 0; i < N; i++)
		positions[i] = heMesh->Vertices().at(i)->position;
	for (size_t i = 0; i < M; i++) {
		auto tri = heMesh->Indices(heMesh->Polygons().at(i));
		indices[3 * i + 0] = static_cast<uint32_t>(tri[0]);
		indices[3 * i + 1] = static_cast<uint32_t>(tri[1]);
		indices[3 * i + 2] = static_cast<uint32_t>(tri[2]);
	}
	//mesh->SetColors({});
	mesh->SetColors({});
	mesh->SetUV({});
	mesh->SetPositions(std::move(positions));
	mesh->SetIndices(std::move(indices));
	mesh->SetSubMeshCount(1);
	mesh->SetSubMesh(0, { 0, M * 3 });
	mesh->GenUV();
	mesh->GenNormals();
	mesh->GenTangents();

	//spdlog::info("HEMesh to Mesh success");
	//spdlog::info("N: {}, M: {}", N, M); 299 562
}

void addNoiseOnHE(std::shared_ptr<HEMeshX> heMesh, float randomScale) {
	if (!heMesh->IsTriMesh()) {
		spdlog::warn("HEMesh isn't triangle mesh");
		return;
	}

	for (auto* v : heMesh->Vertices()) {
		v->position += randomScale * (
			2.f * Ubpa::vecf3{ Ubpa::rand01<float>(),Ubpa::rand01<float>() ,Ubpa::rand01<float>() } - Ubpa::vecf3{ 1.f }
		);
	}

	spdlog::info("Add noise success");
}

void updateColor(std::shared_ptr<Ubpa::Utopia::Mesh> mesh, std::vector<rgbf> colors) {
	mesh->SetToEditable();
	mesh->SetColors(colors);
}

float triArea(pointf3 p0, pointf3 p1, pointf3 p2) {
	auto v0 = p0.as<valf3>() - p1.as<valf3>();
	auto v1 = p2.as<valf3>() - p1.as<valf3>();
	float x = v0.at(1) * v1.at(2) - v0.at(2) * v1.at(1);
	float y = v0.at(2) * v1.at(0) - v0.at(0) * v1.at(2);
	float z = v0.at(0) * v1.at(1) - v0.at(1) * v1.at(0);
	return std::sqrt(x * x + y * y + z * z) / 2;
}

float getA(Vertex* P) {
	 
	//sum of triangle area
	auto getSumArea = [&]() {
		float A = 0;
		for (auto* mesh : P->AdjPolygons()) {
			auto vs = mesh->AdjVertices();
			A += triArea(vs[0]->position, vs[1]->position, vs[2]->position) / 3;
		}
		return A;
	};

	// Voronoi cell
	auto getVoronoiCell = [&]() {
		float A = 0;
		for (auto* e : P->AdjEdges()) {
			auto* Q = e->HalfEdge()->End();
			if (Q == P) Q = e->HalfEdge()->Origin();
			auto* U = e->HalfEdge()->Next()->End();
			auto* V = e->HalfEdge()->Pair()->Next()->End();

			float cot_alpha = (P->position - U->position).cot_theta(Q->position - U->position);
			float cot_beta = (P->position - V->position).cot_theta(Q->position - V->position);

			A += (cot_alpha + cot_beta) * (P->position).distance2(Q->position);
		}
		A = A / 8;
		return A;
	};

	// Mixed Voronoi cell
	auto getMixedVoronoiCell = [&]() {
		float A = 0;
		auto Vs = P->AdjVertices();
		for(size_t i = 0;i < Vs.size();i ++) {
			Vertex* Q = Vs[i];
			Vertex* U = Vs[(i + 1) % Vs.size()];
			/*if (e->HalfEdge()->End() == P) {
				Q = e->HalfEdge()->Origin();
				U = e->HalfEdge()->Next()->End();
			}
			else {
				Q = e->HalfEdge()->End();
				U = e->HalfEdge()->Next()->Origin();
			}*/

			auto is_obtuse = [](Vertex* a, Vertex* b, Vertex* c) {
				return (a->position - b->position).dot(c->position - b->position) < 0.f;
			};

			if (is_obtuse(Q, P, U)) {
				A += triArea(Q->position, P->position, U->position) / 2;
			}
			else if (is_obtuse(Q, U, P) || is_obtuse(U, Q, P)) {
				A += triArea(Q->position, P->position, U->position) / 4;
			}
			else {
				A += (P->position - U->position).cot_theta(Q->position - U->position) * (P->position).distance2(Q->position) / 8;
				A += (P->position - Q->position).cot_theta(U->position - Q->position) * (P->position).distance2(U->position) / 8;
			}
			//spdlog::info("({}, {}, {}) ({}, {}, {}) ({}, {}, {})",
			//	P->position[0], P->position[1], P->position[2],
			//	Q->position[0], Q->position[1], Q->position[2],
			//	U->position[0], U->position[1], U->position[2]);
		}
		return A;
	};

	//return getSumArea();
	//return getVoronoiCell();
	return getMixedVoronoiCell();
}

valf3 getLaplacians(Vertex* P) {
	if (P->IsOnBoundary()) return valf3{ 0.f };

	float A = getA(P);

	//spdlog::info("Area = {}", A);

	valf3 ans{ 0.f };
	for (auto* e : P->AdjEdges()) {
		auto* Q = e->HalfEdge()->End();
		if (Q == P) Q = e->HalfEdge()->Origin();
		auto* U = e->HalfEdge()->Next()->End();
		auto* V = e->HalfEdge()->Pair()->Next()->End();

		float cot_alpha = (P->position - U->position).cot_theta(Q->position - U->position);
		float cot_beta = (P->position - V->position).cot_theta(Q->position - V->position);

		ans += (cot_alpha + cot_beta) * (Q->position - P->position);
		
		//spdlog::info("P = ({}, {}, {}), Q = ({}, {}, {})", 
		//	P->position[0], P->position[1], P->position[2],
		//	Q->position[0], Q->position[1], Q->position[2]);
		//spdlog::info("U = ({}, {}, {}), V = ({}, {}, {})",
		//	U->position[0], U->position[1], U->position[2],
		//	V->position[0], V->position[1], V->position[2]);
		//spdlog::info("cot_alpha = {}, cot_beta = {}", cot_alpha, cot_beta);
	}
	//spdlog::info("ans = ({}, {}, {})", ans[0], ans[1], ans[2]);

	return ans / (2 * A);
}

valf3 getHN(Vertex* P) {
	return -getLaplacians(P) / 2;
}

float getGauss(Vertex* P) {
	if (P->IsOnBoundary()) return 0.f;

	float A = getA(P);

	//spdlog::info("Area = {}", A);

	float ans = 2 * PI<float>;
	for (auto* e : P->AdjEdges()) {
		auto* Q = e->HalfEdge()->End();
		if (Q == P) Q = e->HalfEdge()->Origin();
		auto* U = e->HalfEdge()->Next()->End();

		float cos_theta = (Q->position - P->position).cos_theta(U->position - P->position);

		ans -= acos(cos_theta);

		//spdlog::info("P = ({}, {}, {}), Q = ({}, {}, {})", 
		//	P->position[0], P->position[1], P->position[2],
		//	Q->position[0], Q->position[1], Q->position[2]);
		//spdlog::info("U = ({}, {}, {}), V = ({}, {}, {})",
		//	U->position[0], U->position[1], U->position[2],
		//	V->position[0], V->position[1], V->position[2]);
		//spdlog::info("cot_alpha = {}, cot_beta = {}", cot_alpha, cot_beta);
	}
	//spdlog::info("ans = ({}, {}, {})", ans[0], ans[1], ans[2]);

	return ans / A;
}

rgbf colorMap(float c) {
	float r = 0.8f, g = 1.f, b = 1.f;
	c = c < 0.f ? 0.f : (c > 1.f ? 1.f : c);

	if (c < 1.f / 8.f) {
		r = 0.f;
		g = 0.f;
		b = b * (0.5f + c / (1.f / 8.f) * 0.5f);
	}
	else if (c < 3.f / 8.f) {
		r = 0.f;
		g = g * (c - 1.f / 8.f) / (3.f / 8.f - 1.f / 8.f);
		b = b;
	}
	else if (c < 5.f / 8.f) {
		r = r * (c - 3.f / 8.f) / (5.f / 8.f - 3.f / 8.f);
		g = g;
		b = b - (c - 3.f / 8.f) / (5.f / 8.f - 3.f / 8.f);
	}
	else if (c < 7.f / 8.f) {
		r = r;
		g = g - (c - 5.f / 8.f) / (7.f / 8.f - 5.f / 8.f);
		b = 0.f;
	}
	else {
		r = r - (c - 7.f / 8.f) / (1.f - 7.f / 8.f) * 0.5f;
		g = 0.f;
		b = 0.f;
	}

	return rgbf{ r,g,b };
}

void localRecur(std::shared_ptr<HEMeshX> heMesh, float step) {
	if (!heMesh->IsTriMesh()) {
		spdlog::warn("HEMesh isn't triangle mesh");
		return;
	}

	//for (size_t i = 0; i < mesh->GetPositions().size(); i++)
	//	heMesh->Vertices().at(i)->position = mesh->GetPositions().at(i);

	std::vector<pointf3> pos;

	for (auto* v : heMesh->Vertices()) {
		if (v->IsOnBoundary()) {
			pos.push_back(v->position);
			continue;
		}

		auto Hn = getHN(v);
		pointf3 tmp = v->position.as<valf3>() - step * Hn;
			
		pos.push_back(tmp);
//		spdlog::info("{} : {}, {}, {}", heMesh->Index(v), v->AdjEdges().size(), v->AdjPolygons().size(), v->AdjVertices().size());
		//spdlog::info("({}, {}, {}) {}", Hn[0], Hn[1], Hn[2], step);
		//spdlog::info("({}, {}, {}) -> ({}, {}, {})", v->position[0], v->position[1], v->position[2], tmp[0], tmp[1], tmp[2]);
	}

	int idx = 0;
	for (auto* v : heMesh->Vertices()) {
		v->position = pos[idx ++];
	}

	//spdlog::info("Local recur success");
}

void DenoiseSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<DenoiseData>();
		if (!data)
			return;

		if (ImGui::Begin("Denoise")) {
			/*if (ImGui::Button("Mesh to HEMesh")) {
				meshToHE(data->mesh, data->heMesh);
			}

			if (ImGui::Button("HEMesh to Mesh")) {
				heToMesh(data->heMesh, data->mesh);
			}*/

			if (ImGui::Button("Save Mesh")) {
				data->copy = *data->mesh;
			}

			if (ImGui::Button("Recover Mesh")) {
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}
					if (data->copy.GetPositions().empty()) {
						spdlog::warn("copied mesh is empty");
						return;
					}

					*data->mesh = data->copy;

					spdlog::info("recover success");
				}();
				data->color_type_last = -1;
			}

			if (ImGui::Button("Add Noise")) {
				meshToHE(data->mesh, data->heMesh);
				addNoiseOnHE(data->heMesh, data->randomScale);
				heToMesh(data->heMesh, data->mesh);
				data->color_type_last = -1;
			}

			if (ImGui::Button("Recursion")) {
				meshToHE(data->mesh, data->heMesh);
				for(int i = 0;i < data->num_iterations;i ++)
					localRecur(data->heMesh, data->recurStep);
				heToMesh(data->heMesh, data->mesh);
				data->color_type_last = -1;
			}

			ImGui::RadioButton("Non", &data->color_type, 0); // 
			ImGui::RadioButton("Normal", &data->color_type, 1); // 
			ImGui::RadioButton("Mean Curvature", &data->color_type, 2); // 
			ImGui::RadioButton("Gauss Curvature", &data->color_type, 3); // 

			if (data->color_type_last != data->color_type) {
				
				[&]() {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}
					
					meshToHE(data->mesh, data->heMesh);

					data->color_type_last = data->color_type;

					if (data->color_type == 0) {
						const auto& N = data->mesh->GetNormals().size();
						std::vector<rgbf> colors;
						for (int i = 0; i < N; i ++)
							colors.push_back(valf3{ 1.f });

						updateColor(data->mesh, colors);
						spdlog::info("Set White to Color Success");
					}
					else if (data->color_type == 1) {					
						const auto& normals = data->mesh->GetNormals();
						std::vector<rgbf> colors;
						for (const auto& n : normals)
							colors.push_back((n.as<valf3>() + valf3{ 1.f }) / 2.f);

						updateColor(data->mesh, colors);
						spdlog::info("Set Normal to Color Success");
					}
					else if (data->color_type == 2) {
						std::vector<rgbf> colors;
						for (auto* v : data->heMesh->Vertices())
							colors.push_back(colorMap(getHN(v).norm()));
						updateColor(data->mesh, colors);
						spdlog::info("Set Mean Curvature to Color Success");
					}
					else if (data->color_type == 3) {
						std::vector<rgbf> colors;
						for (auto* v : data->heMesh->Vertices())
							colors.push_back(colorMap(getGauss(v)));
						updateColor(data->mesh, colors);
						spdlog::info("Set Gauss Curvature to Color Success");
					}
				}();

			}



		}
		ImGui::End();
	});
}
