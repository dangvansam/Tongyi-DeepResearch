import re
from typing import Dict, List, Optional, Union
import json5
from qwen_agent.tools.base import BaseToolWithFileAccess, register_tool
from qwen_agent.utils.utils import extract_code
from sandbox_fusion import run_code, RunCodeRequest, RunStatus
from requests.exceptions import Timeout
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

SANDBOX_URL=os.getenv('SANDBOX_URL','')
SANDBOX_FUSION_ENDPOINTS = [
    SANDBOX_URL
]

# Fallback to single endpoint if environment variable exists
if 'SANDBOX_FUSION_ENDPOINT' in os.environ:
    SANDBOX_FUSION_ENDPOINTS = os.environ['SANDBOX_FUSION_ENDPOINT'].split(',')


@register_tool('PythonInterpreter', allow_overwrite=True)
class PythonInterpreter(BaseToolWithFileAccess):
    name = "PythonInterpreter"
    description = 'Execute Python code in a sandboxed environment. Use this to run Python code and get the execution results.\n**Make sure to use print() for any output you want to see in the results.**\nFor code parameters, use placeholders first, and then put the code within <code></code> XML tags, such as:\n<tool_call>\n{"purpose": <detailed-purpose-of-this-tool-call>, "name": <tool-name>, "arguments": {"code": ""}}\n<code>\nHere is the code.\n</code>\n</tool_call>\n'

    parameters = {
        "type": "object",
        "properties": {
            "code": {
                "type": "string",
                "description": "The Python code to execute. Must be provided within <code></code> XML tags. Remember to use print() statements for any output you want to see.",
            }
        },
        "required": ["code"],
    }

    def __init__(self, cfg: Optional[Dict] = None):
        super().__init__(cfg)
        # self.summary_mapping = SummaryMapping()
    
    @property
    def args_format(self) -> str:
        fmt = self.cfg.get('args_format')
        if fmt is None:
            if has_chinese_chars([self.name_for_human, self.name, self.description, self.parameters]):
                fmt = 'The input for this tool should be a Markdown code block.'

            else:
                fmt = 'Enclose the code within triple backticks (`) at the beginning and end of the code.'
        return fmt

    def observation(self, tool: dict, tool_dict: dict, tool_results, empty_mode: bool=False, readpage: bool=False, max_observation_length: int=None, tokenizer=None):
        print('test')
        assert isinstance(tool_results, str), f"result of python code should be str, instead of {type(tool_results)}. {tool_results}"
        return tool_results
    
    @property
    def function(self) -> dict:  
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
        }

    def call(self, params, files= None, timeout = 50, **kwargs) -> str:
        try:
            # super().call(params=params, files=files)  # copy remote files to work_dir
            try:
                code=params.split('<code>')[1].split('</code')[0]
            #     print(params)
            #     if type(params) is str:
            #         params = json5.loads(params)
            #     code = params.get('code', '')
            #     if not code:
            #         code = params.get('raw', '')
            #     triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
            #     if triple_match:
            #         code = triple_match.group(1)
            except Exception:
                return '[Python Interpreter Error]: format error.'

            if not code.strip():
                return '[Python Interpreter Error]: Empty code.'

            # Retry mechanism with randomly sampled endpoints
        #     code=params
            last_error = None
            for attempt in range(8):
                try:
                    # Randomly sample an endpoint for each attempt
                    endpoint = random.choice(SANDBOX_FUSION_ENDPOINTS)
                    print(f"Attempt {attempt + 1}/5 using endpoint: {endpoint}")
                    
                    code_result = run_code(RunCodeRequest(code=code, language='python', run_timeout=timeout), max_attempts=1, client_timeout=timeout, endpoint=endpoint)
                    print("[Python] Code Result", code_result)
                    result = []
                    if code_result.run_result.stdout:
                        result.append(f"stdout:\n{code_result.run_result.stdout}")
                    if code_result.run_result.stderr:
                        result.append(f"stderr:\n{code_result.run_result.stderr}")
                    if code_result.run_result.execution_time >= timeout-1:
                        result.append(f"[PythonInterpreter Error] TimeoutError: Execution timed out.")
                    result = '\n'.join(result)
                    print('SUCCESS RUNNING TOOL')
                    return result if result.strip() else 'Finished execution.'

                except Timeout as e:
                    last_error = f'[Python Interpreter Error] TimeoutError: Execution timed out on endpoint {endpoint}.'
                    print(f"Timeout on attempt {attempt + 1}: {last_error}")
                    if attempt == 4:  # Last attempt
                        return last_error
                    continue
                
                except Exception as e:
                    last_error = f'[Python Interpreter Error]: {str(e)} on endpoint {endpoint}'
                    print(f"Error on attempt {attempt + 1}: {last_error}")
                    if attempt == 4:  # Last attempt
                        return last_error
                    continue

            return last_error if last_error else '[Python Interpreter Error]: All attempts failed.'

        except Exception as e:
            return f"[Python Interpreter Error]: {str(e)}"

    def call_specific_endpoint(self, params: Union[str, dict], endpoint: str, timeout: Optional[int] = 30, **kwargs) -> tuple:
        """Test a specific endpoint directly"""
        try:
            if type(params) is str:
                params = json5.loads(params)
            code = params.get('code', '')
            if not code:
                code = params.get('raw', '')
            triple_match = re.search(r'```[^\n]*\n(.+?)```', code, re.DOTALL)
            if triple_match:
                code = triple_match.group(1)
        except Exception:
            code = extract_code(params)

        if not code.strip():
            return False, '[Python Interpreter Error]: Empty code.'

        try:
            start_time = time.time()
            code_result = run_code(RunCodeRequest(code=code, language='python', run_timeout=timeout), 
                                 max_attempts=1, client_timeout=timeout, endpoint=endpoint)
            end_time = time.time()
            
            result = []
            if code_result.run_result.stdout:
                result.append(f"stdout:\n{code_result.run_result.stdout}")
            if code_result.run_result.stderr:
                result.append(f"stderr:\n{code_result.run_result.stderr}")
            
            result = '\n'.join(result)
            execution_time = end_time - start_time
            return True, result if result.strip() else 'Finished execution.', execution_time

        except Timeout as e:
            return False, f'[Python Interpreter Error] TimeoutError: Execution timed out.', None
        except Exception as e:
            return False, f'[Python Interpreter Error]: {str(e)}', None



def test_single_endpoint(endpoint: str, test_cases: List[dict], timeout: int = 30) -> dict:
    """Test a single endpoint with multiple test cases"""
    executor = PythonInterpreter()
    results = {
        'endpoint': endpoint,
        'status': 'unknown',
        'passed_tests': 0,
        'total_tests': len(test_cases),
        'test_results': [],
        'avg_execution_time': 0,
        'errors': []
    }
    
    execution_times = []
    
    print(f"\n🧪 Testing endpoint: {endpoint}")
    
    for i, test_case in enumerate(test_cases):
        test_name = test_case['name']
        test_code = test_case['code']
        expected_output = test_case.get('expected_output')
        
        print(f"  ├─ Running test {i+1}/{len(test_cases)}: {test_name}")
        
        try:
            success, result, exec_time = executor.call_specific_endpoint(
                {"code": test_code}, endpoint, timeout
            )
            
            test_result = {
                'name': test_name,
                'success': success,
                'result': result,
                'execution_time': exec_time,
                'expected_match': False
            }
            
            if success and exec_time is not None:
                execution_times.append(exec_time)
                
                # Check if output matches expected (if provided)
                if expected_output:
                    # Clean up output for comparison
                    actual_output = result.replace('stdout:\n', '').strip()
                    if expected_output.strip() in actual_output:
                        test_result['expected_match'] = True
                        results['passed_tests'] += 1
                        print(f"  │  ✅ PASSED ({exec_time:.2f}s)")
                    else:
                        print(f"  │  ❌ OUTPUT MISMATCH ({exec_time:.2f}s)")
                        print(f"  │     Expected: {expected_output.strip()}")
                        print(f"  │     Got: {actual_output}")
                else:
                    # No expected output specified, just check if it ran successfully
                    test_result['expected_match'] = True
                    results['passed_tests'] += 1
                    print(f"  │  ✅ PASSED ({exec_time:.2f}s)")
            else:
                print(f"  │  ❌ FAILED: {result}")
                results['errors'].append(f"{test_name}: {result}")
            
            results['test_results'].append(test_result)
            
        except Exception as e:
            print(f"  │  💥 EXCEPTION: {str(e)}")
            results['errors'].append(f"{test_name}: Exception - {str(e)}")
            results['test_results'].append({
                'name': test_name,
                'success': False,
                'result': f"Exception: {str(e)}",
                'execution_time': None,
                'expected_match': False
            })
    
    # Calculate statistics
    if execution_times:
        results['avg_execution_time'] = sum(execution_times) / len(execution_times)
    
    # Determine overall status
    if results['passed_tests'] == results['total_tests']:
        results['status'] = 'healthy'
        print(f"  └─ ✅ ALL TESTS PASSED ({results['passed_tests']}/{results['total_tests']})")
    elif results['passed_tests'] > 0:
        results['status'] = 'partial'
        print(f"  └─ ⚠️  PARTIAL SUCCESS ({results['passed_tests']}/{results['total_tests']})")
    else:
        results['status'] = 'failed'
        print(f"  └─ ❌ ALL TESTS FAILED ({results['passed_tests']}/{results['total_tests']})")
    
    return results


def test_all_endpoints_comprehensive():
    """Comprehensive test suite for all sandbox fusion endpoints"""
    
    # Define comprehensive test cases
    test_cases = [
        {
            'name': 'Basic Math',
            'code': 'print(2 + 2)',
            'expected_output': '4'
        },
        {
            'name': 'String Operations',
            'code': 'print("Hello, " + "World!")',
            'expected_output': 'Hello, World!'
        },
        {
            'name': 'List Operations',
            'code': '''
numbers = [1, 2, 3, 4, 5]
print(f"Sum: {sum(numbers)}")
print(f"Length: {len(numbers)}")
''',
            'expected_output': 'Sum: 15'
        },
        {
            'name': 'Dictionary Operations',
            'code': '''
data = {"name": "Test", "value": 42}
print(f"Name: {data['name']}, Value: {data['value']}")
''',
            'expected_output': 'Name: Test, Value: 42'
        },
        {
            'name': 'Loop and Conditionals',
            'code': '''
result = []
for i in range(5):
    if i % 2 == 0:
        result.append(i)
print(f"Even numbers: {result}")
''',
            'expected_output': 'Even numbers: [0, 2, 4]'
        },
        {
            'name': 'Function Definition',
            'code': '''
def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n-1)

print(f"5! = {factorial(5)}")
''',
            'expected_output': '5! = 120'
        },
        {
            'name': 'Exception Handling',
            'code': '''
try:
    result = 10 / 2
    print(f"Division result: {result}")
except ZeroDivisionError:
    print("Cannot divide by zero")
''',
            'expected_output': 'Division result: 5.0'
        },
        {
            'name': 'Import Standard Library',
            'code': '''
import math
print(f"Pi: {math.pi:.2f}")
print(f"Square root of 16: {math.sqrt(16)}")
''',
            'expected_output': 'Pi: 3.14'
        },
        {
            'name': 'Complex Calculation',
            'code': '''
import math

# Calculate area of a circle
radius = 5
area = math.pi * radius ** 2
print(f"Area of circle with radius {radius}: {area:.2f}")

# Fibonacci sequence
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

fib_10 = fibonacci(10)
print(f"10th Fibonacci number: {fib_10}")
''',
            'expected_output': 'Area of circle with radius 5: 78.54'
        },
        {
            'name': 'Error Handling Test',
            'code': '''
try:
    undefined_variable
except NameError as e:
    print("Caught NameError as expected")
    print("Test passed")
''',
            'expected_output': 'Test passed'
        }
    ]
    
    print("🚀 Starting comprehensive endpoint testing...")
    print(f"📊 Testing {len(SANDBOX_FUSION_ENDPOINTS)} endpoints with {len(test_cases)} test cases each")
    print("=" * 80)
    
    # Test all endpoints concurrently
    all_results = []
    
    # Use ThreadPoolExecutor for concurrent testing
    with ThreadPoolExecutor(max_workers=min(len(SANDBOX_FUSION_ENDPOINTS), 8)) as executor:
        future_to_endpoint = {
            executor.submit(test_single_endpoint, endpoint, test_cases): endpoint 
            for endpoint in SANDBOX_FUSION_ENDPOINTS
        }
        
        for future in as_completed(future_to_endpoint):
            endpoint = future_to_endpoint[future]
            try:
                result = future.result()
                all_results.append(result)
            except Exception as exc:
                print(f'❌ Endpoint {endpoint} generated an exception: {exc}')
                all_results.append({
                    'endpoint': endpoint,
                    'status': 'failed',
                    'passed_tests': 0,
                    'total_tests': len(test_cases),
                    'test_results': [],
                    'avg_execution_time': 0,
                    'errors': [f'Exception during testing: {exc}']
                })
    
    # Print comprehensive summary
    print("\n" + "=" * 80)
    print("📈 COMPREHENSIVE TEST RESULTS SUMMARY")
    print("=" * 80)
    
    healthy_endpoints = [r for r in all_results if r['status'] == 'healthy']
    partial_endpoints = [r for r in all_results if r['status'] == 'partial']
    failed_endpoints = [r for r in all_results if r['status'] == 'failed']
    
    print(f"✅ Healthy endpoints: {len(healthy_endpoints)}/{len(SANDBOX_FUSION_ENDPOINTS)}")
    print(f"⚠️  Partial endpoints: {len(partial_endpoints)}/{len(SANDBOX_FUSION_ENDPOINTS)}")
    print(f"❌ Failed endpoints: {len(failed_endpoints)}/{len(SANDBOX_FUSION_ENDPOINTS)}")
    
    if healthy_endpoints:
        print(f"\n🎉 HEALTHY ENDPOINTS ({len(healthy_endpoints)}):")
        for result in healthy_endpoints:
            print(f"  ✅ {result['endpoint']} (avg: {result['avg_execution_time']:.2f}s)")
    
    if partial_endpoints:
        print(f"\n⚠️  PARTIAL ENDPOINTS ({len(partial_endpoints)}):")
        for result in partial_endpoints:
            print(f"  ⚠️  {result['endpoint']} ({result['passed_tests']}/{result['total_tests']} tests passed)")
    
    if failed_endpoints:
        print(f"\n❌ FAILED ENDPOINTS ({len(failed_endpoints)}):")
        for result in failed_endpoints:
            print(f"  ❌ {result['endpoint']} (0/{result['total_tests']} tests passed)")
            if result['errors']:
                for error in result['errors'][:3]:  # Show first 3 errors
                    print(f"     └─ {error}")
    
    # Performance statistics
    healthy_times = [r['avg_execution_time'] for r in healthy_endpoints if r['avg_execution_time'] > 0]
    if healthy_times:
        print(f"\n⚡ PERFORMANCE STATISTICS:")
        print(f"  Fastest endpoint: {min(healthy_times):.2f}s")
        print(f"  Slowest endpoint: {max(healthy_times):.2f}s")
        print(f"  Average response time: {sum(healthy_times)/len(healthy_times):.2f}s")
    
    # Overall health score
    total_tests = len(SANDBOX_FUSION_ENDPOINTS) * len(test_cases)
    passed_tests = sum(r['passed_tests'] for r in all_results)
    health_score = (passed_tests / total_tests) * 100
    
    print(f"\n🏥 OVERALL SYSTEM HEALTH: {health_score:.1f}%")
    print(f"   ({passed_tests}/{total_tests} total tests passed)")
    
    if health_score >= 90:
        print("   🟢 Excellent - System is performing well")
    elif health_score >= 70:
        print("   🟡 Good - Some endpoints may need attention")
    elif health_score >= 50:
        print("   🟠 Fair - Multiple endpoints having issues")
    else:
        print("   🔴 Poor - System requires immediate attention")
    
    return all_results


def _test():
    """Original test function"""
    # test_code = """import sympy as sp\nX=sp.symbols('X')\npoly_factor = (X**2 - (sp.sqrt(34)+sp.sqrt(14))*X + 2*sp.sqrt(119))*(X**2 - 2*(sp.sqrt(11)+sp.sqrt(6))*X + 4*sp.sqrt(66))\npoly_original = X**4 - sp.sqrt(34)*X**3 - sp.sqrt(14)*X**3 - 2*sp.sqrt(11)*X**3 - 2*sp.sqrt(6)*X**3 + 2*sp.sqrt(374)*X**2 + 2*sp.sqrt(154)*X**2 + 2*sp.sqrt(119)*X**2 + 4*sp.sqrt(66)*X**2 + 4*sp.sqrt(51)*X**2 + 4*sp.sqrt(21)*X**2 - 4*sp.sqrt(1309)*X - 4*sp.sqrt(714)*X - 8*sp.sqrt(561)*X - 8*sp.sqrt(231)*X + 8*sp.sqrt(7854)\nprint('expanded factor matches?', sp.simplify(poly_factor - poly_original) == 0)\nprint('Difference simplified:', sp.simplify(poly_factor - poly_original))\n"""
    # params = {"code": test_code}
    # params = {"code": "x = 1.25\nprint(x*x)"}
    params = "print(1234)"
    # '![fig-001](workspace/tools/CodeInterpreter/ac1b42e5-19fb-460e-b3a3-5f1029658efd.png)'
    executor = PythonInterpreter()
    out = executor.call(params)
    from pprint import pprint
    pprint(out)


if __name__ == '__main__':
    # Run comprehensive endpoint testing
    test_all_endpoints_comprehensive()
    
    print("\n" + "=" * 80)
    print("🔧 Running original test...")
    print("=" * 80)
    
    # Also run the original test
    _test()