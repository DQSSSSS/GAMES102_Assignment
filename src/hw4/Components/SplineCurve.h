#pragma once

#include <UGM/UGM.h>
#include <python.h>

#include <spdlog/spdlog.h>

#include <_deps/imgui/imgui.h>

#include <Eigen/Core>
#include <Eigen/Dense>

#include "tools.h"

using namespace Ubpa;


struct CubicSplineFunc {

	int n, m;
	std::vector<std::vector<float>> a; // m * (n+1) = m * 4

	float x_min, x_max;

	//float get_a(int i, int j) {
	//	DEBUG(0 <= i && i < a.size());
	//	DEBUG(0 <= j && j < a[i].size());
	//	return a[i][j];
	//}

	float get_val(int fi, float t, int order = 0) {
		DEBUG(0 <= fi && fi < m);
		float x;
		if (order == 0) {
			x = a[fi][0] + a[fi][1] * t + a[fi][2] * t * t + a[fi][3] * t * t * t;
			return my_tools::norm_inv({ x }, x_min, x_max)[0];
		}
		else if (order == 1) {
			x = a[fi][1] + 2 * a[fi][2] * t + 3 * a[fi][3] * t * t;
			return x * (x_max - x_min);
		}
		else {
			DEBUG(0);
			exit(0);
		}
		//if (order == 2) y = 2 * a[fi][2] + 6 * a[fi][3] * t;
		
	}

	std::vector<pointf2> norm_x(const std::vector<pointf2>& a) {
		std::vector<float> tmp;
		for (auto p : a) tmp.push_back(p[1]);
		my_tools::pre_norm(tmp, x_min, x_max);
		tmp = my_tools::norm(tmp, x_min, x_max);
		auto ans = a;
		int i = 0;
		for (auto& p : ans) p[1] = tmp[i ++];
		return ans;
	}

	bool init(const std::vector<pointf2>& points0) {
		auto points = norm_x(points0);
		if (points.size() <= 1) {
			spdlog::error("Cubic spline init error: points are too few (" + std::to_string(points.size()) + ").");
			return false;
		}
		m = points.size() - 1;
		n = 3;

		auto get_id = [&](int i, int j) {
			DEBUG(0 <= i && i < m);
			DEBUG(0 <= j && j <= n);
			return i * (n + 1) + j;
		};

		/*std::string info;

		info = "p: ";
		for (int i = 0; i <= m; i++) {
			info += my_tools::get_p_info(points[i]) + " ";
		}
		spdlog::error(info);*/


		Eigen::MatrixXf G(4 * m, 4 * m);
		Eigen::VectorXf Y(4 * m);

		for (int i = 0; i < 4 * m; i++) {
			for (int j = 0; j < 4 * m; j++) {
				G(i, j) = 0;
			}
		}

		/*info = "G:\n";
		for (int i = 0; i < 4 * m; i++) {
			for (int j = 0; j < 4 * m; j++) {
				info += std::to_string(G(i, j)) + " ";
			}
			info += "\n";
		}
		spdlog::error(info);
		spdlog::error("---");*/

		int k = 0;
		for (int i = 0; i <= m; i++) { // m+1 f(ti)=yi
			float t = points[i][0], y = points[i][1];
			if (i < m) {
				G(k, get_id(i, 0)) = 1;
				G(k, get_id(i, 1)) = t;
				G(k, get_id(i, 2)) = t * t;
				G(k, get_id(i, 3)) = t * t * t;
			}
			else {
				G(k, get_id(i - 1, 0)) = 1;
				G(k, get_id(i - 1, 1)) = t;
				G(k, get_id(i - 1, 2)) = t * t;
				G(k, get_id(i - 1, 3)) = t * t * t;
			}
			Y(k) = y;
			k++;
		}

		/*info = "G:\n";
		for (int i = 0; i < 4 * m; i++) {
			for (int j = 0; j < 4 * m; j++) {
				info += std::to_string(G(i, j)) + " ";
			}
			info += "\n";
		}
		spdlog::error(info);
		spdlog::error("---");*/

		for (int i = 1; i < m; i++) { // m-1, f=f
			float t = points[i][0];
			G(k, get_id(i - 1, 0)) = 1;
			G(k, get_id(i - 1, 1)) = t;
			G(k, get_id(i - 1, 2)) = t * t;
			G(k, get_id(i - 1, 3)) = t * t * t;
			G(k, get_id(i, 0)) = -1;
			G(k, get_id(i, 1)) = -t;
			G(k, get_id(i, 2)) = -t * t;
			G(k, get_id(i, 3)) = -t * t * t;
			Y(k) = 0;
			k++;
		}

		/*info = "G:\n";
		for (int i = 0; i < 4 * m; i++) {
			for (int j = 0; j < 4 * m; j++) {
				info += std::to_string(G(i, j)) + " ";
			}
			info += "\n";
		}
		spdlog::error(info);
		spdlog::error("---");*/

		for (int i = 1; i < m; i++) { // m-1, f'=f'
			float t = points[i][0];
			G(k, get_id(i - 1, 1)) = 1;
			G(k, get_id(i - 1, 2)) = 2 * t;
			G(k, get_id(i - 1, 3)) = 3 * t * t;
			G(k, get_id(i, 1)) = -1;
			G(k, get_id(i, 2)) = -2 * t;
			G(k, get_id(i, 3)) = -3 * t * t;
			Y(k) = 0;
			k++;
		}

		/*info = "G:\n";
		for (int i = 0; i < 4 * m; i++) {
			for (int j = 0; j < 4 * m; j++) {
				info += std::to_string(G(i, j)) + " ";
			}
			info += "\n";
		}
		spdlog::error(info);
		spdlog::error("---");*/

		for (int i = 1; i < m; i++) { // m-1, f''=f''
			float t = points[i][0];
			G(k, get_id(i - 1, 2)) = 2;
			G(k, get_id(i - 1, 3)) = 6 * t;
			G(k, get_id(i, 2)) = -2;
			G(k, get_id(i, 3)) = -6 * t;
			Y(k) = 0;
			k++;
		}
		/*
		info = "G:\n";
		for (int i = 0; i < 4 * m; i++) {
			for (int j = 0; j < 4 * m; j++) {
				info += std::to_string(G(i, j)) + " ";
			}
			info += "\n";
		}
		spdlog::error(info);
		spdlog::error("---");*/

		G(k, get_id(0, 2)) = 2;
		G(k, get_id(0, 3)) = 6 * points[0][0];
		Y(k) = 0;
		k++;

		G(k, get_id(m - 1, 2)) = 2;
		G(k, get_id(m - 1, 3)) = 6 * points[m][0];
		Y(k) = 0;
		k++;
		DEBUG(k == 4 * m);

		//spdlog::info("Cubic spline function get G.");

		/*info = "G:\n";
		for (int i = 0; i < 4 * m; i++) {
			for (int j = 0; j < 4 * m; j++) {
				info += std::to_string(G(i, j)) + " ";
			}
			info += "\n";
		}
		spdlog::error(info);
		spdlog::error("---");

		info = "Y:\n";
		for (int i = 0; i < 4 * m; i++) {
			info += std::to_string(Y(i)) + " ";
		}
		spdlog::error(info);*/

		Eigen::VectorXf A = G.colPivHouseholderQr().solve(Y);

		/*info = "sol: ";
		for (int i = 0; i < 4 * m; i++) {
			info += std::to_string(A[i]) + " ";
		}
		spdlog::error(info);

		spdlog::error("---------------------------");*/


		//spdlog::info("Cubic spline function has solved.");

		a.clear(); a.resize(m);
		for (int i = 0; i < m; i++) {
			a[i].resize(n + 1);
			for (int j = 0; j < n + 1; j++) {
				a[i][j] = A[get_id(i, j)];
			}
		}

		/*const float eps = 1e-1;

		for (int i = 0; i <= m; i++) {
			if (i > 0) {
				DEBUG(std::abs(get_val(i - 1, points[i][0]) - points[i][1]) < eps);
				if (std::abs(get_val(i - 1, points[i][0]) - points[i][1]) > eps) {
					spdlog::error("error: {}, {}", get_val(i - 1, points[i][0]), points[i][1]);
				}
			}
			if (i < m) {
				DEBUG(std::abs(get_val(i, points[i][0]) - points[i][1]) < eps);
				if (std::abs(get_val(i, points[i][0]) - points[i][1]) > eps) {
					spdlog::error("error: {}, {}", get_val(i, points[i][0]), points[i][1]);
				}
			}
		}*/
		//for (int i = 0; i <= m; i++) {
		//	if (i > 0) {
		//		spdlog::info("build: {} {} {}", i - 1, get_val(i - 1, points[i][0]), get_val(i - 1, points[i][0], 1));
		//	}
		//	if (i < m) {
		//		spdlog::info("build: {} {} {}", i, get_val(i, points[i][0]), get_val(i, points[i][0], 1));
		//	}
		//}
		//for (int i = 0; i < m; i++) spdlog::info("{}: {} {} {} {}", i, a[i][0], a[i][1], a[i][2], a[i][3]);
		//spdlog::info("------------------");
		//spdlog::info("Cubic spline function init successfully.");
		return true;
	}

	void resolve(int fi, const pointf2& p1, const float& der1, const pointf2& p2, const float& der2) {
		// before norm
		//spdlog::info("{}", fi);
		//spdlog::info("source: ({}, {}) {}  ({}, {}) {}", p1[0], get_val(fi, p1[0]), get_val(fi, p1[0], 1), 
		//	p2[0], get_val(fi, p2[0]), get_val(fi, p2[0], 1));
		//spdlog::info("result: ({}, {}) {}  ({}, {}) {}", p1[0], p1[1], der1, p2[0], p2[1], der2);


		float t1 = p1[0], x1 = p1[1];
		float t2 = p2[0], x2 = p2[1];
		auto tmp = my_tools::norm({ x1, x2 }, x_min, x_max);
		x1 = tmp[0]; x2 = tmp[1];
		float d1 = der1 / (x_max - x_min);
		float d2 = der2 / (x_max - x_min);

		
		Eigen::MatrixXf G(4 , 4);
		Eigen::VectorXf Y(4);
		G(0, 0) = 1;	G(0, 1) = t1;	G(0, 2) = t1 * t1;	G(0, 3) = t1 * t1 * t1;
		G(1, 0) = 1;	G(1, 1) = t2;	G(1, 2) = t2 * t2;	G(1, 3) = t2 * t2 * t2;
		G(2, 0) = 0;	G(2, 1) = 1;	G(2, 2) = 2 * t1;	G(2, 3) = 3 * t1 * t1;
		G(3, 0) = 0;	G(3, 1) = 1;	G(3, 2) = 2 * t2;	G(3, 3) = 3 * t2 * t2;

		Y(0) = x1; 
		Y(1) = x2; 
		Y(2) = d1;  
		Y(3) = d2;

		Eigen::VectorXf A = G.colPivHouseholderQr().solve(Y);

		//spdlog::info("{}: {} {} {} {}", fi, a[fi][0], a[fi][1], a[fi][2], a[fi][3]);
		//spdlog::info("before: {} {}", get_val(fi, p1[0]), p1[1]);
		//spdlog::info("before: {} {}", get_val(fi, p1[0], 1), der1);
		//spdlog::info("before: {} {}", get_val(fi, p2[0]), p2[1]);
		//spdlog::info("before: {} {}", get_val(fi, p2[0], 1), der2);

		a[fi][0] = A[0];
		a[fi][1] = A[1];
		a[fi][2] = A[2];
		a[fi][3] = A[3];

		//spdlog::info("after: {} {}", get_val(fi, p1[0]), p1[1]);
		//spdlog::info("after: {} {}", get_val(fi, p1[0], 1), der1);
		//spdlog::info("after: {} {}", get_val(fi, p2[0]), p2[1]);
		//spdlog::info("after: {} {}", get_val(fi, p2[0], 1), der2);
		//spdlog::info("----------------------");
	}
};

struct CubicSplineCurve {

	CubicSplineFunc x, y;
	std::vector<std::vector<pointf2>> control_points;
	std::vector<pointf2> points;
	std::vector<std::vector<float>> control_points_length;
	std::vector<std::vector<pointf2>> control_points_der;
	std::vector<float> t_vec;

	// float x_min, x_max, y_min, y_max;
	// std::vector<Ubpa::pointf2> points_norm;

	enum POINT_TYPE { SMOOTH, RIGHT, CORNER };
	std::vector<POINT_TYPE> points_type;

	bool is_build_curve = false;
	int focus_cp_i = -1, focus_cp_j = -1;

	static bool is_close(const pointf2& p, const pointf2& o) {
		const float dx = 5;
		const float dy = 5;
		return std::abs(p[0] - o[0]) <= dx && std::abs(p[1] - o[1]) <= dy;
	}

	bool build_curve(const std::vector<pointf2>& points0, const std::vector<float>& t_vec0) {
		points = points0;
		t_vec = t_vec0;
		std::vector<pointf2> x_list, y_list;
		for (size_t i = 0; i < points.size(); i++) {
			x_list.push_back(pointf2(t_vec[i], points[i][0]));
			y_list.push_back(pointf2(t_vec[i], points[i][1]));
		}
		if (!x.init(x_list)) {
			spdlog::error("x init error");
			return false;
		}
		if (!y.init(y_list)) {
			spdlog::error("y init error");
			return false;
		}
		//spdlog::info("Cublic spline function x and y loaded successfully");

		auto get_cp_and_der = [&](int fi, int i, float length) -> std::vector<pointf2> {
			float t = t_vec[i];
			float der_x = x.get_val(fi, t, 1);
			float der_y = y.get_val(fi, t, 1);
			float len = std::sqrt(der_x * der_x + der_y * der_y);
			float ret_x = points[i][0] + der_x / len * length;
			float ret_y = points[i][1] + der_y / len * length;
			return { pointf2(ret_x, ret_y), pointf2(der_x, der_y) };
		};

		int m = points.size() - 1;
		control_points.clear(); control_points.resize(m + 1);
		control_points_length.clear(); control_points_length.resize(m + 1);
		control_points_der.clear(); control_points_der.resize(m + 1);
		for (int i = 0; i <= m; i++) {
			bool flag = false;
			float length;
			if (i > 0) {
				flag = true;
				length = points[i].distance(points[i - 1]) / 3;
				auto tmp = get_cp_and_der(i - 1, i, -length);
				control_points[i].push_back(tmp[0]);
				control_points_der[i].push_back(tmp[1]);
				control_points_length[i].push_back(-length);
			}
			if (i < m) {
				if (!flag) length = points[i].distance(points[i + 1]) / 3;
				auto tmp = get_cp_and_der(i, i, length);
				control_points[i].push_back(tmp[0]);
				control_points_der[i].push_back(tmp[1]);
				control_points_length[i].push_back(length);
			}
		}

		points_type.clear(); points_type.resize(m + 1);
		for (int i = 0; i <= m; i++) {
			points_type[i] = SMOOTH;
		}

		//spdlog::info("Cublic spline curve init successfully");
		is_build_curve = true;
		return true;
	}

	std::vector<pointf2> get_control_points(int id) {
		return control_points[id];
	}

	std::vector<pointf2> get_curve(float lb = 0, float rb = 1, int number = 1000) {
		if (points.size() < 2) return {};
		std::vector<pointf2> ans;
		for (int i = 0, j = 1; i < number; i++) {
			float t = (rb - lb) / number * i + lb;
			while (j < t_vec.size() && t > t_vec[j]) j++; // fi = j-1
			int fi = j - 1;
			ans.push_back(pointf2(x.get_val(fi, t), y.get_val(fi, t)));
		}
		return ans;
	}

	bool found_control_point(const pointf2& o) {
		for (size_t i = 0; i < points.size(); i++) {
			if (is_close(points[i], o)) {
				focus_cp_i = i;
				focus_cp_j = -1;
				return true;
			}
		}

		if (focus_cp_i == -1) return false;

		for (size_t j = 0; j < control_points[focus_cp_i].size(); j++) {
			auto p = control_points[focus_cp_i][j];
			if (is_close(p, o)) {
				focus_cp_j = j;
				return true;
			}
		}
		focus_cp_i = -1;
		focus_cp_j = -1;
		return false;
	}

	bool move_control_point(float dx, float dy) {
		if (focus_cp_i == -1) return false;
		auto cpi = focus_cp_i;
		auto cpj = focus_cp_j;
		if (focus_cp_i != -1 && focus_cp_j != -1) {
			auto cp = control_points[cpi][cpj];
			auto p = points[cpi];
			auto len = control_points_length[cpi][cpj];
			std::vector<pointf2> ncps;

			if (cpi == 0 || cpi + 1 == control_points.size()) { // end point
				ncps.resize(1);
				ncps[cpj] = pointf2(cp[0] + dx, cp[1] + dy);
			}
			else {
				ncps.resize(2);
				ncps[cpj] = pointf2(cp[0] + dx, cp[1] + dy);
				if (points_type[cpi] == SMOOTH) {
					ncps[cpj ^ 1] = pointf2(2 * p[0] - ncps[cpj][0], 2 * p[1] - ncps[cpj][1]);
				}
				else if (points_type[cpi] == RIGHT) {
					float dir_x = p[0] - cp[0], dir_y = p[1] - cp[1];
					float dir_len = p.distance(cp);
					float len = p.distance(control_points[cpi][cpj ^ 1]);
					dir_x = dir_x / dir_len * len;
					dir_y = dir_y / dir_len * len;
					ncps[cpj ^ 1] = pointf2(p[0] + dir_x, p[1] + dir_y);
				}
				else if (points_type[cpi] == CORNER) {
					ncps[cpj ^ 1] = control_points[cpi][cpj ^ 1];
				}
			}
			change_cp(cpi, points[cpi], ncps);
		}
		else if (focus_cp_i != -1) {
			auto np = points[cpi];
			auto ncps = control_points[cpi];
			np = pointf2(np[0] + dx, np[1] + dy);
			for (size_t i = 0; i < ncps.size(); i++) {
				ncps[i] = pointf2(ncps[i][0] + dx, ncps[i][1] + dy);
			}
			//spdlog::info("before change: ({}, {}) ({}, {}) ({}, {})", points[cpi][0], points[cpi][1], control_points[cpi][0][0], control_points[cpi][0][1], control_points[cpi][1][0], control_points[cpi][1][1]);
			change_cp(cpi, np, ncps);
			//spdlog::info("after change: ({}, {}) ({}, {}) ({}, {})", np[0], np[1], ncps[0][0], ncps[0][1], ncps[1][0], ncps[1][1]);
		}
		return true;
	}

	void change_cp(int i, pointf2 np, std::vector<pointf2> ncps) {
		std::vector<float> x_der_vec;
		std::vector<float> y_der_vec;

		/*auto get_cp_and_der = [&](int fi, int i, float length) -> std::vector<pointf2> {
			float t = t_vec[i];
			float der_x = x.get_val(fi, t, 1);
			float der_y = y.get_val(fi, t, 1);
			float ret_x = points[i][0] + der_x / length;
			float ret_y = points[i][1] + der_y / length;
			return { pointf2(ret_x, ret_y), pointf2(der_x, der_y) };
		};*/
		
		auto get_der = [&](int i, int j) -> pointf2 {
			pointf2 p = np;
			pointf2 cp = ncps[j];
			float len = control_points_length[i][j];
			auto dx = cp[0] - p[0];
			auto dy = cp[1] - p[1];
			auto len_now = std::sqrt(dx * dx + dy * dy);
			auto der_x_0 = control_points_der[i][j][0];
			auto der_y_0 = control_points_der[i][j][1];

			float theta = std::atan2(dy, dx);
			float r = len_now / len * std::sqrt(der_x_0 * der_x_0 + der_y_0 * der_y_0);

			auto der_x = r * std::cos(theta);
			auto der_y = r * std::sin(theta);
			
			/*if (dx * der_x < 0) der_x *= -1;
			if (dy * der_y < 0) der_y *= -1;*/
			// TODO
			return pointf2(der_x, der_y);
		};
		
		for (size_t j = 0; j < ncps.size(); j++) {
			auto der = get_der(i, j);
			//spdlog::error("{} {}", der[1], control_points_der[i][j][1]);
			x_der_vec.push_back(der[0]);
			y_der_vec.push_back(der[1]);
		}

		int tot = 0;

		if (i > 0) {
			x.resolve(i - 1, pointf2(t_vec[i], np[0]), x_der_vec[tot],
				pointf2(t_vec[i - 1], points[i - 1][0]), x.get_val(i - 1, t_vec[i - 1], 1));
			y.resolve(i - 1, pointf2(t_vec[i], np[1]), y_der_vec[tot],
				pointf2(t_vec[i - 1], points[i - 1][1]), y.get_val(i - 1, t_vec[i - 1], 1));
			tot++;
		}
		if (i + 1 < points.size()) {
			x.resolve(i, pointf2(t_vec[i], np[0]), x_der_vec[tot],
				pointf2(t_vec[i + 1], points[i + 1][0]), x.get_val(i, t_vec[i + 1], 1));
			y.resolve(i, pointf2(t_vec[i], np[1]), y_der_vec[tot],
				pointf2(t_vec[i + 1], points[i + 1][1]), y.get_val(i, t_vec[i + 1], 1));
		}

		//if (i > 0) {
		//	x.resolve(i - 1, pointf2(t_vec[i], cpi[0]), x_der_times_vec[0],
		//		pointf2(t_vec[i - 1], points[i - 1][0]), x.get_val(i, t_vec[i - 1], 1));
		//	y.resolve(i - 1, pointf2(t_vec[i], cpi[1]), y_der_times_vec[0],
		//		pointf2(t_vec[i - 1], points[i - 1][1]), y.get_val(i, t_vec[i - 1], 1));
		//}

		//if (i + 1 < points.size()) { 
		//	x.resolve(i, pointf2(t_vec[i], cpi[0]), x_der_times_vec,
		//		pointf2(t_vec[i + 1], points[i + 1][0]), x.get_val(i, t_vec[i + 1], 1));
		//	y.resolve(i, pointf2(t_vec[i], cpi[1]), y_der_times_vec,
		//		pointf2(t_vec[i + 1], points[i + 1][1]), y.get_val(i, t_vec[i + 1], 1));
		//}
		
		control_points[i] = ncps;
		points[i] = np;
	}

	void set_point_type(std::string s) {
		if (focus_cp_i == -1) return;
		int cpi = focus_cp_i, cpj = 0;
		auto p = points[cpi], cp = control_points[cpi][cpj];
		auto ncps = control_points[cpi];
		if (cpi > 0 && cpi + 1 < points.size()) {
			if (s == "S") {
				ncps[cpj ^ 1] = pointf2(2 * p[0] - ncps[cpj][0], 2 * p[1] - ncps[cpj][1]);
				points_type[cpi] = SMOOTH;
			}
			else if (s == "R") {
				float dir_x = p[0] - cp[0], dir_y = p[1] - cp[1];
				float dir_len = p.distance(cp);
				float len = p.distance(control_points[cpi][cpj ^ 1]);
				dir_x = dir_x / dir_len * len;
				dir_y = dir_y / dir_len * len;
				ncps[cpj ^ 1] = pointf2(p[0] + dir_x, p[1] + dir_y);
				points_type[cpi] = RIGHT;
			}
			else if (s == "C") {
				ncps[cpj ^ 1] = control_points[cpi][cpj ^ 1];
				points_type[cpi] = CORNER;
			}
		}
		change_cp(cpi, p, ncps);
	}

	int get_focus_point_id() {
		return focus_cp_i;
	}
};

