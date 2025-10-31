# PyVR v0.3.1 Implementation Order

Quick reference for implementing the interface refinements plan.

## Plan Overview

- **Total Phases**: 6
- **Total Lines**: ~4,900 lines of planning documentation
- **Estimated Code**: ~800-1,000 lines of new/modified code
- **Estimated Time**: 2-3 days for experienced developer
- **Test Coverage Target**: >85% for new code, maintain >90% for interface module

## Phase Execution Order

### Phase 1: FPS Counter (566 lines documentation)
**Estimated Time**: 4-6 hours
**Files**: `pyvr/interface/widgets.py`, `pyvr/interface/state.py`, `pyvr/interface/matplotlib_interface.py`
**Tests**: `tests/test_interface/test_fps_counter.py` (14+ tests)
**Complexity**: ⭐️ Low (warmup phase)

**Why First**: Simplest feature, validates performance monitoring approach, independent of other features.

**Key Deliverables**:
- FPSCounter class (~50 lines)
- ImageDisplay enhancements (~80 lines)
- Interface integration (~40 lines)
- Keyboard shortcut 'f'

### Phase 2: Rendering Preset Selector (631 lines documentation)
**Estimated Time**: 5-7 hours
**Files**: `pyvr/interface/widgets.py`, `pyvr/interface/state.py`, `pyvr/interface/matplotlib_interface.py`
**Tests**: `tests/test_interface/test_preset_selector.py` (12+ tests)
**Complexity**: ⭐️⭐️ Medium (widget pattern well-established)

**Why Second**: Builds on widget patterns (ColorSelector), provides immediate user value, foundation for auto-quality in Phase 5.

**Key Deliverables**:
- PresetSelector widget (~100 lines)
- Layout modifications (~50 lines)
- Preset change handler (~60 lines)
- State integration (~30 lines)

### Phase 3: Directional Light Camera Linking (809 lines documentation)
**Estimated Time**: 6-8 hours
**Files**: `pyvr/lighting/light.py`, `pyvr/interface/state.py`, `pyvr/interface/matplotlib_interface.py`
**Tests**: `tests/test_lighting/test_light_linking.py` (20+ tests)
**Complexity**: ⭐️⭐️⭐️ Medium-High (spherical math, careful testing needed)

**Why Third**: Extends lighting system cleanly, can reference FPS counter for validation, independent of histogram complexity.

**Key Deliverables**:
- Light linking methods (~150 lines)
- Interface integration (~80 lines)
- Keyboard shortcut 'l'
- Mathematical correctness validation

### Phase 4: Log-Scale Histogram Background (1,007 lines documentation)
**Estimated Time**: 8-10 hours
**Files**: `pyvr/interface/cache.py` (new), `pyvr/interface/widgets.py`, `pyvr/interface/matplotlib_interface.py`
**Tests**: `tests/test_interface/test_cache.py` (25+ tests), widget tests (4+ tests)
**Complexity**: ⭐️⭐️⭐️⭐️ High (new module, caching infrastructure, hashing)

**Why Fourth**: Most complex feature, benefits from earlier learning, requires careful caching design.

**Key Deliverables**:
- Cache module (~200 lines)
- OpacityEditor enhancements (~100 lines)
- Interface integration (~60 lines)
- Keyboard shortcut 'h'
- Cache validation

### Phase 5: Integration and Polish (784 lines documentation)
**Estimated Time**: 6-8 hours
**Files**: `pyvr/interface/matplotlib_interface.py`, `pyvr/interface/state.py`
**Tests**: `tests/test_interface/test_integration.py` (15+ tests), `tests/test_interface/test_performance.py` (3+ tests)
**Complexity**: ⭐️⭐️⭐️ Medium-High (integration testing, performance validation)

**Why Fifth**: All components exist, now connect them and add polish. Validates feature interactions.

**Key Deliverables**:
- Status display (~80 lines)
- Auto-quality switching (~120 lines)
- Convenience methods (~150 lines)
- Integration tests
- Performance validation

### Phase 6: Documentation Update (874 lines documentation)
**Estimated Time**: 4-6 hours
**Files**: `README.md`, `CLAUDE.md`, `version_notes/v0.3.1_interface_refinements.md`, `example/ModernglRender/v031_features_demo.py`, `tests/README.md`
**Tests**: Manual verification (examples run, docs accurate)
**Complexity**: ⭐️⭐️ Medium (writing, verification)

**Why Last**: All features implemented and tested, can document actual behavior, creates release-ready state.

**Key Deliverables**:
- Updated README (~200 lines added)
- Updated CLAUDE.md (~150 lines added)
- Version notes (~400 lines)
- Feature demo example (~150 lines)
- Test documentation (~100 lines)

## Critical Path Dependencies

```
Phase 1 (FPS) ─┐
               ├─> Phase 5 (Integration) ──> Phase 6 (Docs)
Phase 2 (Preset) ──┤
               ├─> Phase 5 (Integration) ──> Phase 6 (Docs)
Phase 3 (Light) ─┘
               │
Phase 4 (Histogram) ──> Phase 5 (Integration) ──> Phase 6 (Docs)
```

**Phases 1-3**: Can be partially parallelized if multiple developers
**Phase 4**: Independent but benefits from earlier phases
**Phase 5**: Requires Phases 1-4 complete
**Phase 6**: Requires Phase 5 complete

## Testing Checkpoints

After each phase, run:
```bash
# Test new functionality
pytest tests/test_interface/test_[feature].py -v

# Verify no regressions
pytest tests/test_interface/ -v

# Check coverage
pytest --cov=pyvr.interface --cov-report=term-missing tests/test_interface/

# Full regression test (recommended after Phases 3, 5, 6)
pytest tests/ -v
```

## Git Workflow

Each phase has a pre-written commit message. Suggested workflow:

1. Create feature branch: `git checkout -b feature/v0.3.1-interface-refinements`
2. Implement Phase 1, commit with Phase 1 message
3. Implement Phase 2, commit with Phase 2 message
4. Implement Phase 3, commit with Phase 3 message
5. Implement Phase 4, commit with Phase 4 message
6. Implement Phase 5, commit with Phase 5 message
7. Implement Phase 6, commit with Phase 6 message
8. Merge to main: `git checkout main && git merge feature/v0.3.1-interface-refinements`
9. Tag release: `git tag -a v0.3.1 -m "Interface refinements"`
10. Clean up: `rm -rf plans/interface-refinements-v0.3.1/`

## Progress Tracking

Update README.md progress tracker after completing each phase:

```markdown
## Progress Tracker
- [x] Phase 1: FPS Counter Implementation
- [ ] Phase 2: Rendering Preset Selector Widget
- [ ] Phase 3: Directional Light Camera Linking
- [ ] Phase 4: Log-Scale Histogram Background
- [ ] Phase 5: Integration and Polish
- [ ] Phase 6: Documentation Update
```

## Quality Gates

Before proceeding to next phase:
- [ ] All tests passing for current phase
- [ ] Coverage >85% for new code
- [ ] No regressions in existing tests
- [ ] Code formatted (black/isort)
- [ ] Docstrings complete
- [ ] Git commit with phase message

Before finalizing release (after Phase 6):
- [ ] All 330+ tests passing
- [ ] Coverage >90% for interface module
- [ ] All examples run successfully
- [ ] Documentation accurate and complete
- [ ] Version notes finalized
- [ ] README updated
- [ ] CLAUDE.md updated

## Risk Mitigation

**If Phase 4 (Histogram) proves too complex:**
- Can be deferred to v0.3.2
- Phases 1-3 and 5-6 can proceed without it
- Update documentation to mark as "planned for v0.3.2"

**If integration issues arise in Phase 5:**
- Each feature is independently toggleable
- Can disable problematic interactions
- Document known issues in version notes

**If timeline pressure:**
- Phases 1-3 deliver substantial value
- Phase 4 (Histogram) can be v0.3.2
- Phase 5 (Polish) can be streamlined
- Phase 6 (Docs) is essential - don't skip

## Success Metrics

Release is successful if:
1. All acceptance criteria met (see README.md)
2. Zero breaking changes from v0.3.0
3. All features <1% overhead
4. Documentation complete
5. 330+ tests passing
6. User feedback positive

## Files Modified Summary

**New Files** (~10):
- `pyvr/interface/cache.py`
- `tests/test_interface/test_fps_counter.py`
- `tests/test_interface/test_preset_selector.py`
- `tests/test_lighting/test_light_linking.py`
- `tests/test_interface/test_cache.py`
- `tests/test_interface/test_performance.py`
- `version_notes/v0.3.1_interface_refinements.md`
- `example/ModernglRender/v031_features_demo.py`
- `tests/README.md`
- `tmp_dev/histogram_cache/` (directory)

**Modified Files** (~8):
- `pyvr/interface/widgets.py`
- `pyvr/interface/state.py`
- `pyvr/interface/matplotlib_interface.py`
- `pyvr/lighting/light.py`
- `tests/test_interface/test_widgets.py`
- `tests/test_interface/test_integration.py`
- `README.md`
- `CLAUDE.md`

**Total**: ~18 files touched, ~800-1000 lines of code

## Quick Start

To begin implementation:

1. Read this file (you're here!)
2. Read `README.md` for overall context
3. Start with `phase_01.md`
4. Implement each phase sequentially
5. Run tests after each phase
6. Update progress tracker
7. Proceed to next phase

Good luck! The plan is comprehensive and battle-tested.
