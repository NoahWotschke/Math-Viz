#!/bin/bash
export PYTHONPATH="/Users/noahwotschke/Library/Mobile Documents/com~apple~CloudDocs/MathVis/math-viz/mathvis-core:$PYTHONPATH"
cd /Users/noahwotschke/Library/Mobile\ Documents/com~apple~CloudDocs/MathVis/math-viz

echo "════════════════════════════════════════"
echo "TEST SUITE: MathVis Restructuring"
echo "════════════════════════════════════════"

# Test 1: CLI Help
echo ""
echo "✓ TEST 1: CLI Help"
python3 mathvis-cli/solve.py --help > /dev/null 2>&1 && echo "  PASS: Help works" || echo "  FAIL"

# Test 2: CLI with default args (dry run)
echo ""
echo "✓ TEST 2: CLI Default Args (simulation only)"
python3 mathvis-cli/solve.py --seconds 0.1 --skip_error 2>&1 | head -5

# Test 3: CLI with custom Lx
echo ""
echo "✓ TEST 3: CLI with Custom Parameters"
python3 mathvis-cli/solve.py --Lx 2.0 --Ly 1.5 --res 10 --seconds 0.05 --skip_error 2>&1 | head -5

# Test 4: Core imports
echo ""
echo "✓ TEST 4: Core Package Imports"
python3 << 'PYEOF'
try:
    from PDEs.solvers.heat2d_rect import Heat2DRectSolver, Heat2DRectConfig
    from PDEs.domains.rectangle import RectangleDomain
    from PDEs.bc.funcs import const_0, sin_k
    print("  PASS: All core imports work")
except Exception as e:
    print(f"  FAIL: {e}")
PYEOF

# Test 5: Web app imports
echo ""
echo "✓ TEST 5: Web App Imports"
python3 << 'PYEOF'
try:
    import PDEs.vis_settings as vis
    import PDEs.bc.funcs as bc
    from PDEs.domains.rectangle import RectangleDomain
    from PDEs.solvers.heat2d_rect import Heat2DRectSolver, Heat2DRectConfig
    print("  PASS: All web app imports work")
except Exception as e:
    print(f"  FAIL: {e}")
PYEOF

echo ""
echo "════════════════════════════════════════"
echo "ALL TESTS COMPLETED"
echo "════════════════════════════════════════"
