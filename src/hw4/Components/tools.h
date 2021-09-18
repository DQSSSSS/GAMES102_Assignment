#pragma once

#include <UGM/UGM.h>
#include <python.h>

#include <spdlog/spdlog.h>

#include <_deps/imgui/imgui.h>

#include <Eigen/Core>
#include <Eigen/Dense>

using namespace Ubpa;

#define DEBUG(x) ( !(x) && (spdlog::error((std::string("error at line ") + std::string(__FILE__) + " " + std::to_string(__LINE__))), 0) )

struct my_tools {
	static std::vector<float> parameterize(const std::vector<pointf2>& points, int index) {
		if (points.size() == 0) return {};
		if (points.size() == 1) return { 0.5 };
		if (points.size() == 2) return { 0.0f, 1.0f };

		auto len = [&](pointf2 a, pointf2 b) {
			float x = a[0] - b[0], y = a[1] - b[1];
			return std::sqrt(x * x + y * y);
		};
		auto angle = [&](pointf2 a, pointf2 b, pointf2 c) {
			float x1 = a[0] - b[0], y1 = a[1] - b[1];
			float x2 = c[0] - b[0], y2 = c[1] - b[1];
			float tmp = (x1 * x2 + y1 * y2) / len(a, b) / len(b, c);
			return std::acos(tmp);
		};

		int n = points.size();
		std::vector<float> ans;
		if (index == 0) { // uniform
			float d = 1.0 / (n - 1), x = 0;
			for (int i = 0; i < n; i++) {
				ans.push_back(x);
				x += d;
			}
		}
		else if (index == 1) { // chordal
			float sum = 0;
			for (int i = 0; i + 1 < n; i++) {
				float x = len(points[i], points[i + 1]);
				ans.push_back(sum);
				sum += x;
			}
			ans.push_back(sum);
			for (auto& x : ans) {
				x /= sum;
			}
		}
		else if (index == 2) { // centripetal
			float sum = 0;
			for (int i = 0; i + 1 < n; i++) {
				float x = std::sqrt(len(points[i], points[i + 1]));
				ans.push_back(sum);
				sum += x;
			}
			ans.push_back(sum);
			for (auto& x : ans) {
				x /= sum;
			}
		}
		else if (index == 3) { // Foley

			float PI = std::acos(-1);
			auto get = [&](pointf2 a, pointf2 b, pointf2 c) {
				float alpha = std::min(PI - angle(a, b, c), PI / 2);
				return alpha * len(a, b) / (len(a, b) + len(b, c));
			};

			float sum = 0;
			for (int i = 0; i + 1 < n; i++) {
				float r = len(points[i], points[i + 1]);
				float f = 0;
				if (i > 0) {
					f += get(points[i - 1], points[i], points[i + 1]);
				}
				if (i < n - 2) {
					f += get(points[i], points[i + 1], points[i + 2]);
				}

				float tmp = r * (1 + 3.0 / 2 * f);
				ans.push_back(sum);
				sum += tmp;
			}
			ans.push_back(sum);
			for (auto& x : ans) {
				x /= sum;
			}
		}
		DEBUG(ans.size() == n);
		return ans;
	}
	/*
	static std::vector<pointf2> norm(const std::vector<pointf2>& a, float& x_min, float& x_max, float& y_min, float& y_max) {
		auto ans = a;
		x_min = 1e9, y_min = 1e9;
		x_max = -1e9, y_max = -1e9;
		for (size_t i = 0; i < ans.size(); i++) {
			float x = ans[i][0];
			float y = ans[i][1];

			x_min = std::min(x_min, x);
			x_max = std::max(x_max, x);
			y_min = std::min(y_min, y);
			y_max = std::max(y_max, y);
		}

		for (size_t i = 0; i < ans.size(); i++) {
			float& x = ans[i][0];
			float& y = ans[i][1];
			x = (x - x_min) / (x_max - x_min);
			y = (y - y_min) / (y_max - y_min);
		}
		return ans;
	}

	static std::vector<pointf2> norm_inv(const std::vector<pointf2>& a, float x_min, float x_max, float y_min, float y_max) {
		auto ans = a;
		for (size_t i = 0; i < ans.size(); i++) {
			float& x = ans[i][0];
			float& y = ans[i][1];
			x = x * (x_max - x_min) + x_min;
			y = y * (y_max - y_min) + y_min;
		}
		return ans;
	}
	*/

	static void pre_norm(const std::vector<float>& a, float& x_min, float& x_max) {
		x_min = 1e9; x_max = -1e9;
		for (size_t i = 0; i < a.size(); i++) {
			float x = a[i];
			x_min = std::min(x_min, x);
			x_max = std::max(x_max, x);
		}
	}
	static std::vector<float> norm(const std::vector<float>& a, float x_min, float x_max) {
		auto ans = a;
		for (size_t i = 0; i < ans.size(); i++) {
			float& x = ans[i];
			x = (x - x_min) / (x_max - x_min);
		}
		return ans;
	}

	static std::vector<float> norm_inv(const std::vector<float>& a, float x_min, float x_max) {
		auto ans = a;
		for (size_t i = 0; i < ans.size(); i++) {
			float& x = ans[i];
			x = x * (x_max - x_min) + x_min;
		}
		return ans;
	}


	static std::string get_p_info(pointf2 p) {
		return "(" + std::to_string(p[0]) + ", " + std::to_string(p[1]) + ")";
	}


};
