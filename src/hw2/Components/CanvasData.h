#pragma once

#include <UGM/UGM.h>
#include <python.h>

#include <spdlog/spdlog.h>

struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	std::vector<Ubpa::pointf2> points_poly;
	std::vector<Ubpa::pointf2> points_Gauss;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	
	bool opt_enable_lines{ true };
	bool opt_enable_polynomial{ true };
	
	bool opt_enable_Gauss{ true };
	float sigma_Gauss{ 0.1f };

	bool opt_enable_least_squares{ true };
	int n_ls{ 1 };

	bool opt_enable_ridge{ true };
	int n_ridge{ 1 };
	float lambda_ridge{ 0.1f };

	bool opt_enable_rbf{ false };
	int rbf_epoch{ 0 };
	int rbf_n{ 30 };
	bool rbf_is_training{ false };
	float rbf_loss{ 999 };
	PyObject* py_get_new_model;
	PyObject* py_train;
	PyObject* py_test;

	bool is_initialize{ false };

	bool initialize() {

		spdlog::set_pattern("[%L] %v");

		Py_Initialize();

		PyRun_SimpleString("import sys");
		PyRun_SimpleString("sys.path.append('C:/Users/10531/data/Documents/Research/course/games102/homeworks/project/src/hw2')");

		PyObject* pModule = PyImport_ImportModule("rbf_net");
		if (!pModule) {
			spdlog::error("Python module 'rbf_net' not found.");
			return false;
		}

		PyObject* pv = PyObject_GetAttrString(pModule, "get_new_model");
		if (!pv || !PyCallable_Check(pv)) {
			spdlog::error("Function 'get_new_model' is not found in module 'rbf_net'");
			Py_DECREF(pModule);
			return false;
		}
		py_get_new_model = pv;

		pv = PyObject_GetAttrString(pModule, "train_model");
		if (!pv || !PyCallable_Check(pv)) {
			spdlog::error("Function 'train_model' is not found in module 'rbf_net'");
			Py_DECREF(pModule);
			return false;
		}
		py_train = pv;

		pv = PyObject_GetAttrString(pModule, "test_model");
		if (!pv || !PyCallable_Check(pv)) {
			spdlog::error("Function 'test_model' is not found in module 'rbf_net'");
			Py_DECREF(pModule);
			return false;
		}
		py_test = pv;

		spdlog::info("Initialize successfully");
		return true;
	}



	void pushPoint(Ubpa::pointf2 p) {
		int index = points.size();
		for (int i = 0; i < points.size(); i++) {
			if (points[i][0] > p[0]) {
				index = i;
				break;
			}
		}
		points.insert(points.begin() + index, p);
	}
};

#include "details/CanvasData_AutoRefl.inl"