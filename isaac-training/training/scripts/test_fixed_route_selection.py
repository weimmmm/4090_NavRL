import unittest
import torch
from fixed_route_selection import fixed_route_indices, validate_fixed_route_slice


class FixedRouteSelectionTests(unittest.TestCase):
    def test_zero_offset_keeps_existing_evaluations_identical(self):
        ids = torch.tensor([0, 9, 127])
        self.assertTrue(torch.equal(ids, fixed_route_indices(ids, 0)))
        validate_fixed_route_slice(1024, 128, 0)

    def test_holdout_indices_and_partial_reset_use_same_source_rows(self):
        ids = torch.tensor([0, 2, 127])
        rows = fixed_route_indices(ids, 128)
        self.assertTrue(torch.equal(rows, torch.tensor([128, 130, 255])))
        scene = {key:torch.arange(1024)[:, None]+j for j,key in enumerate(
                 ('start_position','goal_position','initial_orientation_wxyz'))}
        saved = {key:value.clone() for key,value in scene.items()}
        for key in scene:
            self.assertTrue(torch.equal(scene[key][rows], saved[key][128:256][ids]))
            self.assertTrue(torch.equal(scene[key], saved[key]))

    def test_out_of_range_and_noninteger_offset_rejected(self):
        for available,count,start in ((1024,128,900),(1024,128,-1),(1024,0,0),
                                       (1024,128,128.5),(1024,128,True)):
            with self.assertRaises(ValueError): validate_fixed_route_slice(available,count,start)


if __name__=='__main__':unittest.main()
