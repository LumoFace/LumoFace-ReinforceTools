import re
import lldb
import json


# workflow
# lldb cmake-build-debug/tests/test_rl_algorithms_td3_second_stage_mlp
# breakpoint set -f include/backprop_tools/containers/operations_generic.h -l 125
# run
# type summary clear
# command script import tools/pretty_print_lldb/matrix.py
# type summary add -F matrix.pretty_print_row_major_alignment -x "^backprop_tools::Matrix<backprop_tools::matrix::Specification<"
# p m1


def decode_row_major(valobj):
    regex = r"^\s*(?:const|\s*)\s*backprop_tools\s*::\s*Matrix\s*<\s*backprop_tools\s*::\s*matrix\s*::\s*Specification\s*<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*backprop_tools\s*::\s*matrix\s*::\s*layouts\s*::\s*RowMajorAlignment\s*<\s*([^,]+)\s*,\s*([^,]+)\s*>\s*,\s*([^,]+)\s*>\s*>\s*(&|\s*)\s*$"
    result = re.match(regex, valobj.type.name)
    if result is None:
        return None
    else:
        meta = dict(zip(["T", "TI", "ROWS", "COLS", "TI2", "ROW_MAJOR_ALIGNMENT", "IS_VIEW"], result.groups()))
        meta["ROWS"] = int(meta["ROWS"])
        meta["COLS"] = int(meta["COLS"])
        meta["ROW_MAJOR_ALIGNMENT"] = int(meta["ROW_MAJOR_ALIGNMENT"])
        ALIGNMENT = meta["ROW_MAJOR_ALIGNMENT"]
        meta["ROW_PITCH"] = ((meta["COLS"] + ALIGNMENT - 1) // ALIGNMENT) * ALIGNMENT
        return meta

def decode_fixed(valobj):
    regex = r"^\s*(?:const|\s*)\s*backprop_tools\s*::\s*Matrix\s*<\s*backprop_tools\s*::\s*matrix\s*::\s*Specification\s*<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*backprop_tools\s*::\s*matrix\s*::\s*layouts\s*::\s*Fixed\s*<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*>\s*,\s*([^,]+)\s*>\s*>\s*(&|\s*)\s*$"
    result = re.match(regex, valobj.type.name)
    if result is None:
        return None
    else:
        meta = dict(zip(["T", "TI", "ROWS", "COLS", "TI2", "ROW_PITCH", "COL_PITCH", "IS_VIEW"], result.groups()))
        meta["ROWS"] = int(meta["ROWS"])
        meta["COLS"] = int(meta["COLS"])
        meta["ROW_PITCH"] = int(meta["ROW_PITCH"])
        meta["COL_PITCH"] = int(meta["COL_PITCH"])
        return meta

def pretty_print_row_major_alignment(valobj, internal_dict, options):
    # regex = r"^\s*(?:const|\s*)\s*backprop_tools\s*::\s*Matrix\s*<\s*backprop_tools\s*::\s*matrix\s*::\s*Specification\s*<\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*([^,]+)\s*,\s*backprop_tools\s*::\s*matrix\s*::\s*layouts\s*::\s*RowMajorAlignment\s*<\s*([^,]+)\s*,\s*([^,]+)\s*>\s*,\s*([^,]+)\s*>\s*>\s*$"
    float_ptr = valobj.GetChildMemberWithName("_data")
    float_type = float_ptr.GetType().GetPointeeType()
    target = valobj.GetTarget()

    meta = decode_row_major(valobj)
    if meta is None:
        return f"Matrix type could not be parsed: {valobj.type.name}"
    else:
        acc = f"{json.dumps(meta)}\n"
        for row_i in range(meta["ROWS"]):
            for col_i in range(meta["COLS"]):
                pos = row_i * meta["ROW_PITCH"] + col_i
                offset = float_ptr.GetValueAsUnsigned() + pos * float_type.GetByteSize()
                val_wrapper = target.CreateValueFromAddress("temp", lldb.SBAddress(