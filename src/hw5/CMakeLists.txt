set(components
  CanvasData
)

set(refls "")

foreach(cmpt ${components})
  set(dst "${CMAKE_CURRENT_SOURCE_DIR}/Components/details/${cmpt}_AutoRefl.inl")
  Ubpa_USReflAutoRefl(
    "${CMAKE_CURRENT_SOURCE_DIR}/Components/${cmpt}.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/Components/details/${cmpt}_AutoRefl.inl"
  )
  list(APPEND refls ${dst})
endforeach()

Ubpa_AddTarget(
  TEST
  SOURCE
    ${refls}
  MODE EXE
  LIB
    Ubpa::Utopia_App_Editor
  INC "${PROJECT_SOURCE_DIR}/include/_deps"
)

target_include_directories(GAMES102_Project_hw5
	PRIVATE $ENV{PYTHONHOME}/include
)
target_link_libraries(GAMES102_Project_hw5
	PRIVATE $ENV{PYTHONHOME}/libs/python38.lib
)