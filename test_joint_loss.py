import unittest
import torch
from m import CompositeRobustLoss


class JointLossTests(unittest.TestCase):
    def test_default_is_arithmetic_mean(self):
        prediction = torch.tensor([[0.2, 0.8]], requires_grad=True)
        target = torch.tensor([[0.7, 0.1]])
        loss = CompositeRobustLoss()
        values = loss.per_task(prediction, target)
        torch.testing.assert_close(loss(prediction, target), values.mean())
        self.assertEqual(loss.task_weights, (0.5, 0.5))

    def test_manuscript_quantile_penalizes_underprediction_more(self):
        prediction = torch.tensor([[0.2, 0.8]])
        target = torch.tensor([[0.8, 0.2]])
        loss = CompositeRobustLoss(weights=(0, 1, 0), tau=0.75,
                                   quantile_mode="formula")
        underprediction, overprediction = loss.per_task(prediction, target)
        torch.testing.assert_close(underprediction, torch.tensor(0.45))
        torch.testing.assert_close(overprediction, torch.tensor(0.15))
        torch.testing.assert_close(loss(prediction, target), torch.tensor(0.30))

    def test_weighted_mean_and_gradient_ratio(self):
        target = torch.tensor([[1.0, 1.0]])
        p = torch.tensor([[0.0, 0.0]], requires_grad=True)
        loss = CompositeRobustLoss(task_weights=(3, 1))
        expected = (3*loss.per_task(p, target)[0]+loss.per_task(p, target)[1])/4
        torch.testing.assert_close(loss(p,target),expected)
        grad, = torch.autograd.grad(loss(p,target),p)
        self.assertAlmostEqual(float(grad[0,0]/grad[0,1]),3.0,places=5)

    def test_both_tasks_backpropagate(self):
        p = torch.tensor([[.1,.2],[.3,.4]],requires_grad=True)
        target = torch.zeros_like(p)
        CompositeRobustLoss(task_weights=(5,1))(p,target).backward()
        self.assertTrue(torch.all(p.grad.abs().sum(0)>0))

    def test_invalid_weights_rejected(self):
        for weights in ((0,0),(-1,2),(1,), (float('nan'),1)):
            with self.subTest(weights=weights), self.assertRaises(ValueError):
                CompositeRobustLoss(task_weights=weights)


if __name__ == '__main__':
    unittest.main(verbosity=2)
