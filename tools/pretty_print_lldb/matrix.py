import re
import lldb
import json


# workflow
# lldb cmake-build-debug/tests/test_rl_algorithms_td3_second_stage_mlp
# breakpoint set -f include/backprop_tools/containers/operations_generic.h -l 125
# run
# type summary clear
# command script import tools/pretty_print_lldb/matrix.py
# type summary add -F matrix.pretty_print_row_major_alignment -x "^backprop_tools::