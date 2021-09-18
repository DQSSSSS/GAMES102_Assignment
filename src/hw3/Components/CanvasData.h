#pragma once

#include <UGM/UGM.h>
#include <python.h>

#include <spdlog/spdlog.h>

#include <_deps/imgui/imgui.h>


struct CanvasData {
	std::vector<Ubpa::pointf2> points;
	std::vector<Ubpa::pointf2> points_input;
	Ubpa::valf2 scrolling{ 0.f,0.f };
	bool opt_enable_grid{ true };
	bool opt_enable_context_menu{ true };
	
	bool opt_enable_lines{ true };
	ImU32 color_lines{ IM_COL32(0, 255, 255, 255) };

	bool opt_enable_polynomial{ true };
	ImU32 color_poly{ IM_COL32(255, 0, 255, 255) };
	
	bool opt_enable_Gauss{ true };
	float sigma_Gauss{ 0.1f };
	ImU32 color_gauss{ IM_COL32(255, 255, 0, 255) };

	bool opt_enable_least_squares{ true };
	int n_ls{ 1 };
	ImU32 color_ls{ IM_COL32(0, 0, 255, 255) };

	bool opt_enable_ridge{ true };
	int n_ridge{ 1 };
	float lambda_ridge{ 0.1f };
	ImU32 color_ridge{ IM_COL32(0, 255, 0, 255) };

	bool opt_enable_rbf{ true };
	int rbf_epoch[2];
	int rbf_n{ 30 };
	bool rbf_is_training{ false };
	float rbf_loss[2];
	PyObject* py_get_new_model;
	PyObject* py_models[2];
	PyObject* py_train;
	PyObject* py_test;
	ImU32 color_rbf{ IM_COL32(255, 0, 0, 255) };

	bool is_initialize{ false };

	bool opt_enable_params{ false };
	int params_indix{ 0 };
	
	int debug_val{ 0 };

	bool initialize() {
		rbf_loss[0] = rbf_loss[1] = -1;
		spdlog::set_pattern("[%L] %v");

		Py_Initialize();

		PyRun_SimpleString("import sys");
		PyRun_SimpleString("sys.path.append('C:/Users/10531/data/Documents/Research/course/games102/homeworks/project/src/hw3/')");

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

		PyObject* args = PyTuple_New(1);
		PyTuple_SetItem(args, 0, PyLong_FromLong(30));
		
		py_models[0] = PyObject_CallObject(py_get_new_model, args);
		if (!py_models[0]) {
			spdlog::error("Model 0 loaded error");
			return false;
		}
		Py_INCREF(py_models[0]);

		py_models[1] = PyObject_CallObject(py_get_new_model, args);
		if (!py_models[1]) {
			spdlog::error("Model 1 loaded error");
			return false;
		}
		Py_INCREF(py_models[1]);

		Py_DECREF(args);
		spdlog::info("Initialize successfully");
		return true;
	}

	void pushPoint(Ubpa::pointf2 p) {
		points_input.push_back(p);
		int index = points.size();
		for (int i = 0; i < points.size(); i++) {
			if (points[i][0] > p[0]) {
				index = i;
				break;
			}
		}
		points.insert(points.begin() + index, p);
	}

	void clearPoints() {
		points.clear();
		points_input.clear();
	}
};

#include "details/CanvasData_AutoRefl.inl"