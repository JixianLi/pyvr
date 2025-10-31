# PyVR Test Suite

Comprehensive test suite for PyVR volume rendering toolkit.

## Test Structure

```
tests/
├── test_camera/              # Camera system tests (42 tests)
├── test_config.py            # RenderConfig tests (33 tests)
├── test_lighting/            # Lighting system tests (42 tests)
│   ├── test_light.py         # Light class tests
│   └── test_light_linking.py # Camera-linked lighting (v0.3.1)
├── test_transferfunctions/   # Transfer function tests (36 tests)
├── test_moderngl_renderer/   # OpenGL rendering tests (71 tests)
├── test_interface/           # Interactive interface tests (137 tests)
│   ├── test_state.py         # State management
│   ├── test_widgets.py       # Widget components
│   ├── test_matplotlib.py    # Main interface
│   ├── test_fps_counter.py   # FPS counter (v0.3.1)
│   ├── test_preset_selector.py # Preset selector (v0.3.1)
│   ├── test_cache.py         # Histogram caching (v0.3.1)
│   ├── test_integration.py   # Feature integration
│   └── test_bug_fixes.py     # Phase 5.5 bug fixes (v0.3.1)
└── test_volume/              # Volume data tests
```

## Running Tests

### Full Test Suite
```bash
pytest tests/
```

### By Module
```bash
pytest tests/test_interface/  # Interface tests
pytest tests/test_camera/     # Camera tests
pytest tests/test_lighting/   # Lighting tests
```

### With Coverage
```bash
pytest --cov=pyvr --cov-report=html tests/
```

### v0.3.1 Specific Tests
```bash
# FPS counter tests
pytest tests/test_interface/test_fps_counter.py

# Preset selector tests
pytest tests/test_interface/test_preset_selector.py

# Camera-linked lighting tests
pytest tests/test_lighting/test_light_linking.py

# Histogram caching tests
pytest tests/test_interface/test_cache.py

# Integration tests
pytest tests/test_interface/test_integration.py

# Bug fix tests
pytest tests/test_interface/test_bug_fixes.py
```

## Test Categories

### Unit Tests
Test individual components in isolation using mocks.

### Integration Tests
Test interaction between components.

### Bug Fix Tests (v0.3.1)
Validate critical bug fixes from Phase 5.5:
- Status display overlay fix
- Light linking error handling
- Matplotlib keybinding conflicts
- Mouse event handlers
- Auto-quality rendering during drag

## Coverage Targets

- New code: >85% coverage
- Interface module: >90% coverage
- Overall project: ~86% coverage

## Mocking Strategy

OpenGL tests use mock-based approach for CI/CD compatibility:
- `tests/test_moderngl_renderer/conftest.py`: Central mock fixtures
- Allows testing without display server or GPU
- Mocks `moderngl.create_context()`, texture creation, shaders

## Current Test Count

**Total**: 361 tests
- v0.3.1 added: +77 tests
  - FPS counter: 14 tests
  - Preset selector: 13 tests
  - Light linking: 20 tests
  - Histogram caching: 14 tests
  - Histogram widget: 4 tests
  - Integration: 10 tests
  - Bug fixes: 8 tests
- All passing ✓

## Test Breakdown by Module

| Module | Tests | Coverage |
|--------|-------|----------|
| Camera System | 42 | 95-97% |
| RenderConfig | 33 | 100% |
| Lighting System | 42 | 99-100% |
| Transfer Functions | 36 | 88-100% |
| ModernGL Renderer | 71 | 93-98% |
| Interface | 137 | >90% |
| Volume & Datasets | - | 56-93% |
| **Total** | **361** | **~86%** |

## Contributing Tests

When adding features:
1. Write unit tests for new classes/methods
2. Add integration tests for feature interactions
3. Include edge cases and error conditions
4. Aim for >85% coverage
5. Ensure all existing tests still pass

## Testing v0.3.1 Features

### FPS Counter
```bash
pytest tests/test_interface/test_fps_counter.py -v
```
Tests rolling average calculation, display toggle, reset functionality.

### Quality Presets
```bash
pytest tests/test_interface/test_preset_selector.py -v
```
Tests preset switching, validation, UI updates.

### Camera-Linked Lighting
```bash
pytest tests/test_lighting/test_light_linking.py -v
```
Tests light position updates, offset calculations, link/unlink operations.

### Histogram Caching
```bash
pytest tests/test_interface/test_cache.py -v
```
Tests hash generation, cache hit/miss, persistence, speedup validation.

### Integration
```bash
pytest tests/test_interface/test_integration.py -v
```
Tests feature interactions, workflows, concurrent operations.

### Bug Fixes
```bash
pytest tests/test_interface/test_bug_fixes.py -v
```
Tests Phase 5.5 critical bug fixes.

## Continuous Integration

All tests run in CI with:
- Python 3.11+
- Mock-based OpenGL (no display server required)
- Coverage reporting
- All 361 tests must pass

## Performance Tests

Some tests validate performance characteristics:
- FPS counter overhead: <1%
- Light linking overhead: <1%
- Histogram cache speedup: >5x
- Render throttling: ~100ms intervals

## Known Test Limitations

1. **Interface tests**: Some interactive features difficult to test without GUI
2. **Performance tests**: May vary on different hardware
3. **Mock-based tests**: Don't catch actual OpenGL driver issues

## Test Development Guidelines

- Use descriptive test names: `test_<feature>_<condition>_<expected>`
- Add docstrings for complex tests
- Use fixtures from `conftest.py`
- Mock external dependencies
- Test edge cases and error conditions
- Keep tests fast (<1s per test)

---

**PyVR Test Suite** - Comprehensive testing for reliable volume rendering! ✅
