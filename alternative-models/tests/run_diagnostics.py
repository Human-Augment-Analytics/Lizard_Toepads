import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import test_common
import test_stacked_hourglass
import test_vit
import test_hrnet

def main():
    print("Running alternative-models diagnostics...\n")

    suites = [
        ("common", test_common.run),
        ("stacked-hourglass", test_stacked_hourglass.run),
        ("vit", test_vit.run),
        ("hrnet", test_hrnet.run),
    ]

    total_passed = 0
    total_failed = 0

    for name, run_fn in suites:
        try:
            p, f = run_fn()
        except Exception as e:
            print(f"  [ERROR] {name} suite crashed: {e}")
            p, f = 0, 1
        total_passed += p
        total_failed += f
        print()

    print("=" * 40)
    print(f"TOTAL: {total_passed} passed, {total_failed} failed")
    print("=" * 40)

    sys.exit(0 if total_failed == 0 else 1)

if __name__ == "__main__":
    main()
