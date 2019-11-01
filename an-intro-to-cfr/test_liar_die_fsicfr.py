import liar_die_fsicfr as ld
import pytest
import numpy as np


class TestLiarDieTrainer:
    def test_init(self):
        n_sides = 6

        trainer = ld.LiarDieTrainer(n_sides)

        assert trainer.n_sides == n_sides
        assert len(trainer.response_nodes) == n_sides + 1
        assert len(trainer.claim_nodes) == n_sides 

        for arr in trainer.response_nodes:
            assert len(arr) == n_sides + 1

        for arr in trainer.claim_nodes:
            assert len(arr) == n_sides + 1

        # check special cases
        for prev_claim in range(1, n_sides):
            # max claim has only 1 legal action: DOUBT
            final_node = trainer.response_nodes[prev_claim][n_sides]
            assert isinstance(final_node, ld.Node)
            assert final_node.n_actions == 1

    def test_get_init_rolls(self):
        n_sides = 6
        trainer = ld.LiarDieTrainer(n_sides)

        roll = trainer.get_init_rolls()
        assert roll.shape == (n_sides, )
        assert (roll <= n_sides).all()

    def test_visit_response_nodes_fwd(self):
        n_sides = 6
        trainer = ld.LiarDieTrainer(n_sides)
        roll = np.asarray([1 for _ in range(n_sides + 1)])
        trainer.init_start_prob(roll)

        # all rolls are 1
        # check that initial claim node start with reach probabilities 1
        init_claim_node = trainer.claim_nodes[0][roll[0]]
        assert init_claim_node.reach_player == 1.
        assert init_claim_node.reach_opponent == 1.

        # check that claim nodes start with zero reach probabilities
        opp_claim = 1
        next_claim_node = trainer.claim_nodes[opp_claim][roll[opp_claim]]
        assert next_claim_node.reach_player == 0.
        assert next_claim_node.reach_opponent == 0.
        
        # since the response_nodes[my_claim][opp_claim] have 0 reach probabilities for
        # all my_claim values, the next_node would not have accumulated any probabilities
        trainer.visit_response_nodes_fwd(opp_claim, roll)
        assert next_claim_node.reach_player == 0.
        assert next_claim_node.reach_opponent == 0.

    def test_visit_claim_nodes_fwd(self):
        n_sides = 6
        trainer = ld.LiarDieTrainer(n_sides)
        roll = np.asarray([1 for _ in range(n_sides + 1)])
        trainer.init_start_prob(roll)

        # all rolls are 1
        # check that initial claim node start with reach probabilities 1
        init_claim_node = trainer.claim_nodes[0][roll[0]]
        assert init_claim_node.reach_player == 1.
        assert init_claim_node.reach_opponent == 1.

        # check claim_node characteristics
        # initial claim, so no prior opponent claim
        opp_claim = 0
        next_roll = roll[opp_claim]
        claim_node = trainer.claim_nodes[opp_claim][next_roll]
        assert isinstance(claim_node, ld.Node)
        assert claim_node.reach_opponent == 1.
        assert claim_node.reach_player == 1.

        action_prob = claim_node.get_strategy()
        # all claims should be legal
        assert len(action_prob) == n_sides
        # should be uniform over all legal actions since no strategy has been learnt yet
        assert np.isclose(action_prob, 1. / n_sides).all()

        for my_claim in range(opp_claim + 1, n_sides + 1):
            next_claim_prob = action_prob[my_claim - opp_claim - 1]
            assert np.isclose(next_claim_prob, 1./n_sides)

            response_node = trainer.response_nodes[opp_claim][my_claim]
            assert isinstance(response_node, ld.Node)
            assert response_node.reach_player == 0.
            assert response_node.reach_opponent == 0.

        trainer.visit_claim_nodes_fwd(opp_claim, roll)

        # check that accumulation did occur
        for my_claim in range(opp_claim + 1, n_sides + 1):
            response_node = trainer.response_nodes[opp_claim][my_claim]
            assert isinstance(response_node, ld.Node)
            assert np.isclose(response_node.reach_player, 1)
            assert np.isclose(response_node.reach_opponent, 1./n_sides)

