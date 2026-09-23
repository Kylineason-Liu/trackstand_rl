import unittest

import numpy as np

from nominal_bike_control import NominalBikeController, ScaleBikeModel


class NominalBikeControllerTest(unittest.TestCase):
    def setUp(self):
        self.model = ScaleBikeModel(dt=0.02, forward_speed=2.0)
        self.system = self.model.sys

    def test_repeated_poles_and_speed_scheduling(self):
        controller = NominalBikeController(
            self.system, method="place_multiple_poles", wc=-5.0
        )
        np.testing.assert_allclose(
            np.sort_complex(controller.closed_loop_poles),
            np.array([-5.0, -5.0, -5.0]),
            atol=8e-5,
        )

        old_gain = controller.K.copy()
        self.model.updateSysParam(2.7, min_forw_vel=1.0)
        controller.updateSysAndGain(self.model.sys)
        self.assertFalse(np.allclose(controller.K, old_gain))
        np.testing.assert_allclose(
            np.sort_complex(controller.closed_loop_poles),
            np.array([-5.0, -5.0, -5.0]),
            atol=8e-5,
        )

    def test_nominal_equilibrium_satisfies_model(self):
        controller = NominalBikeController(
            self.system, method="place_multiple_poles", wc=-5.0
        )
        reference = np.array([[0.1]])
        x_eq, u_eq = controller.nominal_equilibrium(reference)
        np.testing.assert_allclose(
            controller.A @ x_eq + controller.Bu @ u_eq, 0.0, atol=1e-10
        )
        np.testing.assert_allclose(controller.Co @ x_eq, reference, atol=1e-10)

        equilibrium = controller.solve_equilibrium(reference)
        self.assertTrue(equilibrium.is_unique)
        self.assertLess(equilibrium.residual_norm, 1e-10)
        # A non-zero steering equilibrium requires a non-zero lean angle.
        self.assertGreater(abs(equilibrium.state[1, 0]), 1e-6)

    def test_inconsistent_full_state_reference_is_rejected(self):
        controller = NominalBikeController(
            self.system, method="place_multiple_poles", wc=-5.0
        )
        naive_reference = np.array([[0.1], [0.0], [0.0]])
        with self.assertRaisesRegex(ValueError, "not a steady state"):
            controller.validate_state_reference(naive_reference)

        equilibrium = controller.solve_equilibrium(np.array([[0.1]]))
        state_ref, control_ref = controller.validate_state_reference(
            equilibrium.state, equilibrium.control
        )
        np.testing.assert_allclose(state_ref, equilibrium.state)
        np.testing.assert_allclose(control_ref, equilibrium.control)

    def test_feedback_uses_measurement_directly_and_applies_limits(self):
        controller = NominalBikeController(
            self.system,
            method="place_multiple_poles",
            wc=-5.0,
            u_min=-0.2,
            u_max=0.2,
        )
        state = np.array([[0.0], [0.2], [0.0]])
        command = controller.stepAndGetControl(np.array([[0.0]]), state, 1.0)
        expected_raw = -controller.K @ state
        np.testing.assert_allclose(controller.last_unsaturated_control, expected_raw)
        np.testing.assert_allclose(command, np.clip(expected_raw, -0.2, 0.2))
        self.assertFalse(hasattr(controller, "observer"))


if __name__ == "__main__":
    unittest.main()
