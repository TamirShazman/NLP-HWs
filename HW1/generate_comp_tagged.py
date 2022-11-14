from inference import tag_all_test
import pickle

if __name__ == '__main__':
    part1_weights_path = ''
    part1_comp_path = 'data/comp1.words'
    part1_output_path = 'comp_m1_123456789_987654321.wtag'

    part2_weights_path = ''
    part2_comp_path = 'data/comp2.words'
    part2_output_path = 'comp_m1_123456789_987654321.wtag'

    # load part one weights with feature2id
    with open(part1_comp_path, 'rb') as f:
        optimal_params1, feature2id1 = pickle.load(f)
    # tag part one
    tag_all_test(part1_comp_path, optimal_params1[0], feature2id1, part1_output_path)

    # load part two weights with feature2id
    with open(part1_comp_path, 'rb') as f:
        optimal_params2, feature2id2 = pickle.load(f)
    # tag part two
    tag_all_test(part2_comp_path, optimal_params2[0], feature2id2, part2_output_path)