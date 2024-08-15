"""utils"""
def execute(prog, state_dict=None, get_traceback=False):
    try:
        exec(prog, state_dict)
    except Exception as e:
        import traceback
        return traceback.format_exc() if get_traceback else e
        return trace
import pandas as pd
# def exec_and_show(code, df):
#     code, fig, err = execute_code(
#         code=code,#new_code,#output_code,
#         vars={'df':df, 'pd':pd}
#     )
#     fig.show() if fig else print(f"err: {err}")

def execute_code(code, vars=None, output_suggestions=[], output_transforms=[]):
    print("executing code in PoT module")
    print("var dict (keys): ", vars.keys())
    if not code.strip():
        return code, None, "Error: Empty code before execution."
    
    result = execute(prog=code, state_dict=vars)
    if result:# Error
        return code, None, str(result)
    
    # if result:# Error
    #     return code, None, str(result), None if get_traceback else code, None, str(result)
    
    # running exec stores local vars in the sent state dictionary
    output = vars.get('fig')#'generated_code')
    print("output from exec: ", output)

    if not output: # program did not store output in generated_code
        # return code, None, "Error: Output not stored in `generated_code`. Make sure the final output is in a variable called `generated_code`"
        return code, None, "Error: Output not stored in a variable called `fig`. Make sure the final plot is stored in a variable called `fig`."

    # Perform suggestions and transformations on output
    [suggestion for suggestion in output_suggestions]
    for f in output_transforms:
        output = f(output)

    return code, output, None

def execute_code_dict(code, vars=None, output_suggestions=[], output_transforms=[], get_traceback=False):
    print("executing code in PoT module")
    print("var dict (keys): ", vars.keys())
    print("code is:==", code, "==", code=='')
    if not code or code == '' or code.strip() == '':
        return {
            'code':code,
            'err':"Error: Empty code before execution.",
            'fig':None
        }#result
        # return code, None, "Error: Empty code before execution."
    
    err = execute(prog=code, state_dict=vars, get_traceback=get_traceback)

    # result = {'code':code, 'err':err, }
    if err:# Error
        return {
            'code':code,
            'err':err,
            'fig':None
        }#result
        
    # running exec stores local vars in the sent state dictionary
    fig = vars.get('fig')#'generated_code')
    
    print("output from exec: ", fig)

    if not fig: # program did not store output in generated_code
        return {'code':code, 'fig':None, 'err':"Error: Output not stored in a variable called `fig`. Make sure the final plot is stored in a variable called `fig`."}

    # Perform suggestions and transformations on output
    [suggestion for suggestion in output_suggestions]
    for f in output_transforms:
        fig = f(fig)

    return {
            'code':code,
            'err':None,
            'fig':fig
        }#code, fig, None