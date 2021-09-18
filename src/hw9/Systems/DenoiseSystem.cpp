#include "DenoiseSystem.h"

#include "../Components/DenoiseData.h"

#include <_deps/imgui/imgui.h>

#include <spdlog/spdlog.h>

#include <Eigen/core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>

#include "QEM.h"

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

void heToMesh(std::shared_ptr<HEMeshX> heMesh, std::vector<uint32_t> indices, std::shared_ptr<Ubpa::Utopia::Mesh> mesh) {
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
	assert(indices.size() % 3 == 0);
	const size_t M = indices.size() / 3;
	std::vector<Ubpa::pointf3> positions(N);
	for (size_t i = 0; i < N; i++)
		positions[i] = heMesh->Vertices().at(i)->position;
	//mesh->SetColors({});
	mesh->SetColors(std::vector<rgbf>(N, valf3{1, 1, 1}));
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
		for (size_t i = 0; i < Vs.size(); i++) {
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

			if (triArea(Q->position, P->position, U->position) < 1e-4) continue;

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

	return ans / (2.f * A);
}

valf3 getHN(Vertex* P) {
	return -getLaplacians(P) / 2.f;
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
		v->position = pos[idx++];
	}

	//spdlog::info("Local recur success");
}

enum ParamShape{ NON, SQUARE, CIRCLE };
enum ParamMethod{ UNIFORM, DISTANCE};

std::vector<pointf3> parametrization(std::shared_ptr<HEMeshX> heMesh, ParamShape shape, ParamMethod method, float radius) {
	std::vector<pointf3> ans;

	if (shape == ParamShape::NON) {
		for (auto* v : heMesh->Vertices()) {
			if (v->IsOnBoundary()) {
				ans.push_back(v->position);
			}
			else {
				ans.push_back(pointf3(0, 0, 0));
			}
		}
		return ans;
	}

	for (auto* v : heMesh->Vertices()) 
		ans.push_back(pointf3(0, 0, 0));

	Vertex* p = NULL;
	for (auto* v : heMesh->Vertices()) {
		if (v->IsOnBoundary()) {
			p = v;
			break;
		}
	}
	if (p == NULL) {
		spdlog::warn("No Boundary");
		return ans;
	}
	std::vector<Vertex*> boundaries;
	while (true) {
		boundaries.push_back(p);
		int n = boundaries.size();
		Vertex* nx = NULL;
		for (auto* v : p->AdjVertices()) {
			if (v->IsOnBoundary()) {
				if (n >= 2 && v == boundaries[n-2]) continue;
				nx = v; 
				break;
			}
		}
		if (nx == boundaries[0]) break;
		p = nx;
	}

	std::vector<float> pos_linear;
	if (method == ParamMethod::UNIFORM) {
		for (size_t i = 0; i < boundaries.size(); i++) {
			pos_linear.push_back(1.0 * i / boundaries.size());
		}
	}
	else if (method == ParamMethod::DISTANCE) {
		for (size_t i = 0; i < boundaries.size(); i++) {
			if (i == 0) pos_linear.push_back(0);
			else pos_linear.push_back(boundaries[i]->position.distance(boundaries[i - 1]->position) + pos_linear.back());
		}
		for (size_t i = 0; i < pos_linear.size(); i++) {
			pos_linear[i] /= pos_linear.back();
		}
	}

	auto posOnCircle = [](const std::vector<float>& pos_linear, float r) {
		std::vector<pointf3> ans;
		for (auto pl : pos_linear) {
			float alpha = pl * 2 * PI<float>;
			float x = std::sin(alpha) * r;
			float y = std::cos(alpha) * r;
			ans.push_back(pointf3(x, y, 0));
		}
		return ans;
	};

	auto posOnSquare = [](const std::vector<float>& pos_linear, float r) {
		std::vector<pointf3> ans;
		float t = 0;
		for (auto pl : pos_linear) {
			t = pl;
			float x, y;
			if (t < 1.0 / 4)	  x = 1,		  y = 8 * t - 1;
			else if (t < 2.0 / 4) x = -8 * t + 3, y = 1;
			else if (t < 3.0 / 4) x = -1,		  y = -8 * t + 5;
			else				  x = 8 * t - 7,  y = -1;
			x = x * r;
			y = y * r;
			//spdlog::info("{}: ({}, {})", t, x, y);
			ans.push_back(pointf3(x, y, 0));
		}
		return ans;
	};

	std::vector<pointf3> pos;
	if (shape == ParamShape::CIRCLE) pos = posOnCircle(pos_linear, radius);
	else if (shape == ParamShape::SQUARE) pos = posOnSquare(pos_linear, radius);
	
	for (size_t i = 0; i < boundaries.size();i ++) {
		auto* v = boundaries[i];
		auto position = pos[i];
		ans[heMesh->Index(v)] = position;
	}
	spdlog::info("Parametrization success");
	return ans;
}

enum GlobalType{ CONSTANT, COT };

void globalMinizing(std::shared_ptr<HEMeshX> heMesh, std::vector<pointf3> boundaries_pos, GlobalType gt) {
	auto& Vs = heMesh->Vertices();
	int totalPoints = Vs.size();

	Eigen::SparseMatrix<float> A(totalPoints, totalPoints);
	for (size_t i = 0; i < Vs.size(); i++) {
		if (Vs[i]->IsOnBoundary()) {
			A.coeffRef(i, i) = 1;
		}
		else {
			auto& P = Vs[i];
			//for (auto* e : P->AdjEdges()) {
			//	auto* Q = e->HalfEdge()->End();
			//	if (Q == P) Q = e->HalfEdge()->Origin();
			//	auto* U = e->HalfEdge()->Next()->End();
			//	auto* V = e->HalfEdge()->Pair()->Next()->End();
			/*[&]() {
				
				auto is_same = [&](std::vector<Vertex*> a, std::vector<Vertex*> b) {
					if (a.size() != b.size()) return false;
					if (a.size() == 0) return true;
					for(size_t i = 0;i < b.size();i ++) 
						if (a[0] == b[i]) {
							for (size_t j = 0; j < a.size(); j++) {
								if (a[j] != b[(i + j) % b.size()]) {
									return false;
								}
							}
						}
					return true;
				};

				std::vector<Vertex*> vec_true;
				auto now = P->AdjVertices()[0];
				while (true) {
					vec_true.push_back(now);
					auto is_in_P_adj = [&](Vertex* v) {
						for (auto x : P->AdjVertices()) if (x == v) return true;
						return false;
					};
					int n = vec_true.size();
					Vertex* nx = NULL;
					for (auto v : now->AdjVertices()) {
						if (is_in_P_adj(v)) {
							if (n >= 2 && vec_true[n - 2] == v) continue;
							nx = v; break;
						}
					}
					if (nx == vec_true[0]) break;
					now = nx;
				}

				auto vec_true_rev = vec_true;
				std::reverse(vec_true_rev.begin(), vec_true_rev.end());

				auto vec1 = P->AdjVertices();
				std::vector<Vertex*> vec2;
				for (auto* e : P->AdjEdges()) {
					auto* Q = e->HalfEdge()->End();
					if (Q == P) Q = e->HalfEdge()->Origin();
					vec2.push_back(Q);
				}
				if (!is_same(vec_true, vec1) && !is_same(vec_true_rev, vec1)) spdlog::info("error on 1");
				if (!is_same(vec_true, vec2) && !is_same(vec_true_rev, vec2)) spdlog::info("error on 2");
			}();*/

			auto adjVs = P->AdjVertices();
			if (gt == GlobalType::COT) {
				A.coeffRef(i, i) = 0;
				for (size_t k = 0; k < adjVs.size(); k++) {
					Vertex* Q = adjVs[k];
					Vertex* U = adjVs[(k + 1) % adjVs.size()];
					Vertex* V = adjVs[(k - 1 + adjVs.size()) % adjVs.size()];

					int j = heMesh->Index(Q);

					float cot_alpha = (Q->position - U->position).cot_theta(P->position - U->position);
					float cot_beta = (Q->position - V->position).cot_theta(P->position - V->position);
					float wij = cot_alpha + cot_beta;
					A.coeffRef(i, j) = -wij;
					A.coeffRef(i, i) += wij;
				}
			}
			else if (gt == GlobalType::CONSTANT) {
				A.coeffRef(i, i) = adjVs.size();
				for (size_t k = 0; k < adjVs.size(); k++) {
					int j = heMesh->Index(adjVs[k]);
					A.coeffRef(i, j) = -1;
				}
			}
			else {
				spdlog::warn("This global type is undefined");
				continue;
			}
		}
	}

	auto solve = [&](int dim) {
		Eigen::SparseVector<float> b(totalPoints);
		for (size_t i = 0; i < Vs.size(); i++) {
			if (Vs[i]->IsOnBoundary()) {
				b.coeffRef(i) = boundaries_pos[i][dim];
			}
			else {
				b.coeffRef(i) = 0;
			}
		}
		Eigen::ConjugateGradient<Eigen::SparseMatrix<float> > solver;
		solver.setTolerance(1e-4);
		solver.compute(A.transpose() * A);
		Eigen::SparseVector<float> x = solver.solve(A.transpose() * b);
		std::vector<float> ans;
		for (size_t i = 0; i < Vs.size(); i++) ans.push_back(x.coeff(i));
		return ans;
	};

	for (int dim : {0, 1, 2}) {
		auto ans = solve(dim);
		for (size_t i = 0; i < Vs.size(); i++)
			Vs[i]->position[dim] = ans[i];
	}
	return;
}

void DenoiseSystem::OnUpdate(Ubpa::UECS::Schedule& schedule) {
	schedule.RegisterCommand([](Ubpa::UECS::World* w) {
		auto data = w->entityMngr.GetSingleton<DenoiseData>();
		if (!data)
			return;

		if (ImGui::Begin("Denoise")) {
			
			static QuadricSimplification qem;

			if (ImGui::Button("Initialization")) {
				[&] {
					if (!data->mesh) {
						spdlog::warn("mesh is nullptr");
						return;
					}
					data->copy = *data->mesh;
					meshToHE(data->mesh, data->heMesh);
					qem.init(data->heMesh);
					data->rate = 1;
					spdlog::info("init success");
				}();
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
					meshToHE(data->mesh, data->heMesh);
					qem.init(data->heMesh);
					data->rate = 1;
					spdlog::info("recover success");
				}();
			}

			data->rate = std::min(data->rate, 1.f);

			if (qem.hasInitialized()) {
				if (qem.trans(data->rate)) {
					heToMesh(data->heMesh, qem.getTriangles(), data->mesh);
				}

				ImGui::Text("N: %d", data->heMesh->Vertices().size());
				ImGui::Text("M: %d", qem.getTriangles().size()/3);

				ImGui::SliderFloat("rate", &data->rate, 0.f, 1.f);
			}

		}
		ImGui::End();
		});
}
