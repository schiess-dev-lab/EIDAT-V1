import math
import sys
import tempfile
import unittest
from pathlib import Path


APP_ROOT = Path(__file__).resolve().parents[1]
if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))


from ui_next import backend  # noqa: E402

try:
    from openpyxl import load_workbook
except Exception:  # pragma: no cover - optional dependency guard for local runs
    load_workbook = None  # type: ignore[assignment]


def _have_scipy() -> bool:
    try:
        import scipy  # noqa: F401
    except Exception:
        return False
    return True


def _fit_points(count: int = 6, *, input_2: bool = False) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []
    for index in range(count):
        point: dict[str, object] = {
            "observation_id": f"obs-{index + 1}",
            "run_name": f"Run{index + 1}",
            "serial": f"SN-{index % 2 + 1:03d}",
            "condition_label": f"Cond{index + 1}",
            "display_name": f"Cond{index + 1}",
            "input_1": float(index + 1),
            "actual_mean": 2.0 * float(index + 1) + 1.0,
            "sample_count": 3,
            "xfit": [0.0] * 50,  # bulky key that must NOT survive serialization on models only
        }
        if input_2:
            point["input_2"] = float(index) * 0.5
        points.append(point)
    return points


def _solver_result(
    *,
    output_target: str,
    output_selector: str = "",
    input1_target: str,
    input1_selector: str = "",
    input2_target: str = "",
    input2_selector: str = "",
    fit_family: str = "polynomial",
    master_model: dict[str, object] | None = None,
    three_sigma_offset: float = 0.5,
    three_sigma_support_rows: list[dict[str, object]] | None = None,
) -> dict[str, object]:
    if master_model is None:
        master_model = {
            "fit_family": fit_family,
            "coeffs": [2.0, 1.0],
            "normalize_x": False,
            "xfit": [0.0] * 100,
            "yfit": [0.0] * 100,
        }
    return {
        "fit_family": str(master_model.get("fit_family") or fit_family),
        "master_model": master_model,
        "equation": f"{output_target} = f({input1_target})",
        "x_norm_equation": "",
        "output_target": output_target,
        "output_units": "u",
        "output_selector": output_selector,
        "input1_target": input1_target,
        "input1_units": "u",
        "input1_selector": input1_selector,
        "input2_target": input2_target,
        "input2_units": "u" if input2_target else "",
        "input2_selector": input2_selector,
        "input3_target": "",
        "input3_units": "",
        "input3_selector": "",
        "uses_control_period": False,
        "uses_staged_mediator": False,
        "run_type_mode": "steady_state",
        "three_sigma_basis": "population",
        "three_sigma_offset": three_sigma_offset,
        "three_sigma_support_rows": list(three_sigma_support_rows or []),
        "fit_points": _fit_points(input_2=bool(input2_target)),
        "solver_branch": str(master_model.get("fit_family") or fit_family),
        "selection_reason": "test",
        "rmse": 0.123456,
        "run_count": 3,
        "serial_count": 2,
        "sample_count": 18,
        "candidate_scores": {"polynomial": 1.0},
        "slice_rows": [{"bulk": True}],
    }


def _pchip_master_model(scale: float = 1.0) -> dict[str, object]:
    knots = [float(i) for i in range(12)]
    values = [scale * math.sqrt(v + 1.0) for v in knots]
    return {
        "fit_family": backend.TD_PERF_FIT_MODE_MONOTONE_PCHIP,
        "params": {
            "knots": knots,
            "knot_values": values,
            "left_y": values[0],
            "right_y": values[-1],
        },
    }


class TestSmartSolverSavedEquationStore(unittest.TestCase):
    def test_saved_entry_round_trip_and_trimming(self) -> None:
        result = _solver_result(
            output_target="Prop Per Pulse",
            output_selector="canonical:prop_per_pulse",
            input1_target="PF",
            input1_selector="canonical:pf",
            input2_target="Pulse Width",
            input2_selector="canonical:pulse_width",
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            entry = backend.td_smart_solver_build_saved_equation_entry(
                project_dir / "cache.sqlite3",
                name="Prop per pulse",
                result=result,
                config={"run_type_mode": "steady_state"},
                plot_metadata={"run_selection_label": "Smart Equation Solver"},
            )
            backend.td_smart_solver_upsert_saved_equation(project_dir, entry)
            store = backend.load_td_saved_smart_equations(project_dir)

        entries = store.get("entries") or []
        self.assertEqual(len(entries), 1)
        loaded = entries[0]
        self.assertEqual(loaded.get("name"), "Prop per pulse")
        self.assertEqual(loaded.get("output_selector"), "canonical:prop_per_pulse")
        self.assertEqual(
            loaded.get("input_selectors", {}).get("input_1", {}).get("selector"),
            "canonical:pf",
        )
        loaded_result = loaded.get("result") or {}
        model = loaded_result.get("master_model") or {}
        self.assertNotIn("xfit", model)
        self.assertNotIn("yfit", model)
        self.assertNotIn("candidate_scores", loaded_result)
        self.assertNotIn("slice_rows", loaded_result)
        fit_points = loaded_result.get("fit_points") or []
        self.assertTrue(fit_points)
        for point in fit_points:
            self.assertNotIn("xfit", point)
            self.assertLessEqual(
                set(point.keys()),
                set(backend.TD_SMART_SOLVER_SAVED_FIT_POINT_KEYS),
            )

    def test_upsert_replaces_by_name_and_delete(self) -> None:
        result = _solver_result(output_target="ISP", input1_target="Prop Per Pulse")
        with tempfile.TemporaryDirectory() as tmpdir:
            project_dir = Path(tmpdir)
            first = backend.td_smart_solver_build_saved_equation_entry(
                project_dir / "cache.sqlite3", name="ISP", result=result
            )
            backend.td_smart_solver_upsert_saved_equation(project_dir, first)
            second = backend.td_smart_solver_build_saved_equation_entry(
                project_dir / "cache.sqlite3", name="isp", result=result
            )
            replaced = backend.td_smart_solver_upsert_saved_equation(project_dir, second)
            store = backend.load_td_saved_smart_equations(project_dir)
            self.assertEqual(len(store.get("entries") or []), 1)
            self.assertEqual(replaced.get("id"), first.get("id"))
            self.assertEqual(replaced.get("saved_at"), first.get("saved_at"))
            self.assertTrue(backend.td_smart_solver_delete_saved_equation(project_dir, first.get("id")))
            store = backend.load_td_saved_smart_equations(project_dir)
            self.assertEqual(store.get("entries"), [])

    def test_normalize_rejects_unexportable_entry(self) -> None:
        result = _solver_result(output_target="X", input1_target="Y")
        result["master_model"] = {"fit_family": "not_a_family"}
        self.assertIsNone(
            backend._td_smart_solver_normalize_saved_entry(
                {"name": "Bad", "result": result}
            )
        )


class TestSmartSolverEquationChain(unittest.TestCase):
    def _entry(self, name: str, result: dict[str, object]) -> dict[str, object]:
        return backend.td_smart_solver_build_saved_equation_entry(
            Path("missing.sqlite3"), name=name, result=result
        )

    def test_chain_selector_match_and_order(self) -> None:
        isp = self._entry(
            "ISP",
            _solver_result(
                output_target="ISP",
                output_selector="canonical:isp",
                input1_target="Prop Per Pulse",
                input1_selector="canonical:prop_per_pulse",
            ),
        )
        prop = self._entry(
            "Prop Per Pulse",
            _solver_result(
                output_target="Prop Per Pulse",
                output_selector="canonical:prop_per_pulse",
                input1_target="PF",
                input1_selector="canonical:pf",
                input2_target="Pulse Width",
                input2_selector="canonical:pulse_width",
            ),
        )
        chain = backend.td_smart_solver_build_equation_chain([isp, prop])
        self.assertEqual(chain["order"], [str(prop["id"]), str(isp["id"])])
        self.assertEqual(len(chain["links"]), 1)
        link = chain["links"][0]
        self.assertEqual(link["downstream_id"], str(isp["id"]))
        self.assertEqual(link["upstream_id"], str(prop["id"]))
        self.assertEqual(link["variable_key"], "input_1")

    def test_chain_display_name_fallback_case_insensitive(self) -> None:
        upstream = self._entry(
            "Up", _solver_result(output_target="Prop Per Pulse", input1_target="PF")
        )
        downstream = self._entry(
            "Down", _solver_result(output_target="ISP", input1_target="prop per pulse")
        )
        chain = backend.td_smart_solver_build_equation_chain([downstream, upstream])
        self.assertEqual(len(chain["links"]), 1)
        self.assertEqual(chain["order"], [str(upstream["id"]), str(downstream["id"])])

    def test_chain_unmatched_input_stays_manual(self) -> None:
        a = self._entry("A", _solver_result(output_target="Out A", input1_target="In A"))
        b = self._entry("B", _solver_result(output_target="Out B", input1_target="In B"))
        chain = backend.td_smart_solver_build_equation_chain([a, b])
        self.assertEqual(chain["links"], [])
        self.assertEqual(chain["order"], [str(a["id"]), str(b["id"])])

    def test_chain_self_link_ignored_with_warning(self) -> None:
        node = self._entry("Self", _solver_result(output_target="Same", input1_target="Same"))
        chain = backend.td_smart_solver_build_equation_chain([node])
        self.assertEqual(chain["links"], [])
        self.assertTrue(any("self-link" in text for text in chain["warnings"]))

    def test_chain_multiple_upstreams_into_multi_input_node(self) -> None:
        prop = self._entry(
            "Prop", _solver_result(output_target="Prop Per Pulse", input1_target="PF")
        )
        width = self._entry(
            "Width", _solver_result(output_target="Pulse Width", input1_target="Freq")
        )
        isp = self._entry(
            "ISP",
            _solver_result(
                output_target="ISP",
                input1_target="Prop Per Pulse",
                input2_target="Pulse Width",
            ),
        )
        chain = backend.td_smart_solver_build_equation_chain([isp, prop, width])
        keys = {(link["variable_key"], link["upstream_id"]) for link in chain["links"]}
        self.assertEqual(
            keys,
            {("input_1", str(prop["id"])), ("input_2", str(width["id"]))},
        )
        self.assertEqual(chain["order"][-1], str(isp["id"]))

    def test_chain_duplicate_producers_warns_first_wins(self) -> None:
        first = self._entry(
            "First", _solver_result(output_target="Shared", input1_target="X1")
        )
        second = self._entry(
            "Second", _solver_result(output_target="Shared", input1_target="X2")
        )
        consumer = self._entry(
            "Consumer", _solver_result(output_target="Y", input1_target="Shared")
        )
        chain = backend.td_smart_solver_build_equation_chain([first, second, consumer])
        link = next(l for l in chain["links"] if l["downstream_id"] == str(consumer["id"]))
        self.assertEqual(link["upstream_id"], str(first["id"]))
        self.assertTrue(any("multiple equations" in text for text in chain["warnings"]))

    def test_chain_cycle_raises_with_names(self) -> None:
        a = self._entry("Alpha", _solver_result(output_target="A Out", input1_target="B Out"))
        b = self._entry("Beta", _solver_result(output_target="B Out", input1_target="A Out"))
        with self.assertRaises(RuntimeError) as ctx:
            backend.td_smart_solver_build_equation_chain([a, b])
        message = str(ctx.exception)
        self.assertIn("cycle", message)
        self.assertIn("Alpha", message)
        self.assertIn("Beta", message)


@unittest.skipIf(load_workbook is None, "openpyxl unavailable")
class TestSmartSolverMasterWorkbook(unittest.TestCase):
    def _entry(self, name: str, result: dict[str, object]) -> dict[str, object]:
        return backend.td_smart_solver_build_saved_equation_entry(
            Path("missing.sqlite3"), name=name, result=result
        )

    def test_master_workbook_chained_structure(self) -> None:
        prop = self._entry(
            "Prop Per Pulse",
            _solver_result(
                output_target="Prop Per Pulse",
                output_selector="canonical:prop_per_pulse",
                input1_target="PF",
                input2_target="Pulse Width",
            ),
        )
        isp = self._entry(
            "ISP",
            _solver_result(
                output_target="ISP",
                input1_target="Prop Per Pulse",
                input1_selector="canonical:prop_per_pulse",
            ),
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "master.xlsx"
            backend.td_smart_solver_export_master_workbook(
                Path(tmpdir) / "cache.sqlite3",
                out_path,
                entries=[isp, prop],
            )
            wb = load_workbook(str(out_path), data_only=False)
            try:
                self.assertEqual(wb.sheetnames[0], "Master Calculator")
                fit_sheets = [name for name in wb.sheetnames if name.startswith("Fit Points E")]
                self.assertEqual(len(fit_sheets), 2)
                self.assertEqual(wb.sheetnames.count("Model Parameters"), 1)
                self.assertEqual(wb.sheetnames.count("Model Support"), 1)

                ws = wb["Master Calculator"]
                cells = {
                    (cell.row, cell.column): cell.value
                    for row in ws.iter_rows()
                    for cell in row
                    if cell.value not in (None, "")
                }
                text_values = [str(v) for v in cells.values()]
                # topological order: prop block first
                e1_title = next(v for v in text_values if str(v).startswith("Equation 1:"))
                self.assertIn("Prop Per Pulse", e1_title)
                e2_title = next(v for v in text_values if str(v).startswith("Equation 2:"))
                self.assertIn("ISP", e2_title)

                # defined names are per-block
                self.assertIsNotNone(wb.defined_names.get("SmartSolver_E1_input_1"))
                self.assertIsNotNone(wb.defined_names.get("SmartSolver_E2_input_1"))

                # chained input: the E2 input_1 cell contains a formula pointing
                # at E1's pred_mean cell
                e2_ref = wb.defined_names.get("SmartSolver_E2_input_1").attr_text
                cell_ref = e2_ref.split("!")[-1].replace("$", "")
                col_letter = "".join(ch for ch in cell_ref if ch.isalpha())
                row_num = int("".join(ch for ch in cell_ref if ch.isdigit()))
                chained_value = str(ws[f"{col_letter}{row_num}"].value or "")
                self.assertTrue(chained_value.startswith("="))
                # the referenced cell holds E1's pred_mean formula
                target_ref = chained_value[1:].replace("$", "")
                target_formula = str(ws[target_ref].value or "")
                self.assertIn("SmartSolver_E1_input_1", target_formula)

                self.assertIn("Chain Map", text_values)
                # every formula stays within Excel's limit
                formulas = [
                    str(cell.value)
                    for sheet in wb.worksheets
                    for row in sheet.iter_rows()
                    for cell in row
                    if isinstance(cell.value, str) and cell.value.startswith("=")
                ]
                self.assertTrue(formulas)
                self.assertLessEqual(max(len(f) for f in formulas), 8192)
            finally:
                wb.close()

    @unittest.skipUnless(_have_scipy(), "scipy not installed")
    def test_master_workbook_two_pchip_equations_no_name_collisions(self) -> None:
        first = self._entry(
            "PCHIP One",
            _solver_result(
                output_target="Mid Value",
                input1_target="Raw Input",
                master_model=_pchip_master_model(1.0),
                three_sigma_support_rows=[
                    {"control_period": 10.0, "three_sigma_offset": 0.4},
                    {"control_period": 20.0, "three_sigma_offset": 0.6},
                ],
            ),
        )
        second = self._entry(
            "PCHIP Two",
            _solver_result(
                output_target="Final Value",
                input1_target="Mid Value",
                master_model=_pchip_master_model(2.0),
                three_sigma_support_rows=[
                    {"control_period": 10.0, "three_sigma_offset": 0.7},
                    {"control_period": 20.0, "three_sigma_offset": 0.9},
                ],
            ),
        )
        for entry in (first, second):
            entry_result = entry.get("result") or {}
            entry_result["uses_control_period"] = True
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "master_pchip.xlsx"
            backend.td_smart_solver_export_master_workbook(
                Path(tmpdir) / "cache.sqlite3",
                out_path,
                entries=[second, first],
            )
            wb = load_workbook(str(out_path), data_only=False)
            try:
                names = list(wb.defined_names.keys())
                self.assertEqual(len(names), len(set(names)))
                self.assertTrue(any(name.startswith("SmartSolver_E1_PCHIP_") for name in names))
                self.assertTrue(any(name.startswith("SmartSolver_E2_PCHIP_") for name in names))
                self.assertTrue(any(name.startswith("SmartSolver_E1_ThreeSigma_") for name in names))
                self.assertTrue(any(name.startswith("SmartSolver_E2_ThreeSigma_") for name in names))
                self.assertIn("Formula Support", wb.sheetnames)
                self.assertNotIn("Formula Support1", wb.sheetnames)
                self.assertEqual(wb["Formula Support"].sheet_state, "hidden")
            finally:
                wb.close()


if __name__ == "__main__":
    unittest.main()
