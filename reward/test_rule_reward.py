# Copyright (c) 2026
#
# reward/rule_reward.py 的单元测试
#
# 运行方式：
#   cd train && python -m pytest reward/test_rule_reward.py -v
#   或者不依赖 pytest，直接: python reward/test_rule_reward.py

import unittest

from rule_reward import (
    RuleReward,
    check_format,
    compute_score,
    is_equivalent,
    normalize_answer,
)


class TestFormatCheck(unittest.TestCase):
    def test_strict_format_ok(self):
        text = "<think>因为 1+1=2</think><answer>2</answer>"
        result = check_format(text)
        self.assertTrue(result.strict_ok)
        self.assertTrue(result.has_think)
        self.assertTrue(result.has_answer)
        self.assertEqual(result.answer_text, "2")

    def test_strict_format_with_surrounding_whitespace(self):
        text = "  \n<think>推理过程</think>\n<answer>42</answer>\n  "
        result = check_format(text)
        self.assertTrue(result.strict_ok)
        self.assertEqual(result.answer_text, "42")

    def test_missing_think_tag(self):
        text = "答案是 <answer>2</answer>"
        result = check_format(text)
        self.assertFalse(result.strict_ok)
        self.assertFalse(result.has_think)
        self.assertTrue(result.has_answer)
        self.assertEqual(result.answer_text, "2")

    def test_missing_answer_tag(self):
        text = "<think>推理过程</think>最终答案是2"
        result = check_format(text)
        self.assertFalse(result.strict_ok)
        self.assertTrue(result.has_think)
        self.assertFalse(result.has_answer)
        self.assertIsNone(result.answer_text)

    def test_extra_text_breaks_strict_but_loose_extracts_last_answer(self):
        text = "<think>step1</think><answer>1</answer> 额外多余文本 <answer>2</answer>"
        result = check_format(text)
        self.assertFalse(result.strict_ok)
        # 宽松抽取取最后一个 <answer> 块
        self.assertEqual(result.answer_text, "2")

    def test_no_tags_at_all(self):
        text = "这里没有任何标签，直接说答案是2"
        result = check_format(text)
        self.assertFalse(result.strict_ok)
        self.assertFalse(result.has_think)
        self.assertFalse(result.has_answer)
        self.assertIsNone(result.answer_text)


class TestNormalizeAnswer(unittest.TestCase):
    def test_boxed_extraction(self):
        self.assertEqual(normalize_answer("\\boxed{42}"), "42")

    def test_plain_number(self):
        self.assertEqual(normalize_answer("42"), "42")

    def test_comma_thousand_separator(self):
        self.assertEqual(normalize_answer("1,234"), "1234")

    def test_fraction_shorthand(self):
        # \frac12 -> \frac{1}{2}
        self.assertEqual(normalize_answer("\\frac12"), "\\frac{1}{2}")

    def test_half_equivalence_string(self):
        self.assertEqual(normalize_answer("0.5"), "\\frac{1}{2}")

    def test_slash_fraction(self):
        self.assertEqual(normalize_answer("3/4"), "\\frac{3}{4}")

    def test_degree_symbol_removed(self):
        self.assertEqual(normalize_answer("90^\\circ"), "90")

    def test_percent_removed(self):
        self.assertEqual(normalize_answer("50\\%"), "50")


class TestIsEquivalent(unittest.TestCase):
    def test_exact_match(self):
        self.assertTrue(is_equivalent("42", "42"))

    def test_boxed_vs_plain(self):
        self.assertTrue(is_equivalent("\\boxed{42}", "42"))

    def test_fraction_forms_equivalent(self):
        self.assertTrue(is_equivalent("1/2", "\\frac{1}{2}"))
        self.assertTrue(is_equivalent("0.5", "\\frac{1}{2}"))

    def test_thousand_separator_equivalent(self):
        self.assertTrue(is_equivalent("1,234", "1234"))

    def test_wrong_answer(self):
        self.assertFalse(is_equivalent("41", "42"))

    def test_none_inputs(self):
        self.assertFalse(is_equivalent(None, "42"))
        self.assertFalse(is_equivalent("42", None))

    def test_numeric_tolerance(self):
        # 浮点数值近似相等（如四舍五入误差）
        self.assertTrue(is_equivalent("3.14159", "3.14159"))


class TestRuleRewardScoring(unittest.TestCase):
    def setUp(self):
        self.reward = RuleReward(
            correct_score=1.0,
            wrong_score=0.0,
            format_bonus=0.1,
            format_penalty=-0.1,
            no_answer_score=-1.0,
        )

    def test_correct_and_well_formatted(self):
        text = "<think>1+1=2</think><answer>2</answer>"
        result = self.reward.score(text, "2")
        self.assertTrue(result.correct)
        self.assertTrue(result.format_ok)
        self.assertAlmostEqual(result.score, 1.1)

    def test_correct_but_bad_format(self):
        text = "答案是 <answer>2</answer>，没有think标签"
        result = self.reward.score(text, "2")
        self.assertTrue(result.correct)
        self.assertFalse(result.format_ok)
        self.assertAlmostEqual(result.score, 0.9)  # 1.0 - 0.1

    def test_wrong_but_well_formatted(self):
        text = "<think>算错了</think><answer>3</answer>"
        result = self.reward.score(text, "2")
        self.assertFalse(result.correct)
        self.assertTrue(result.format_ok)
        self.assertAlmostEqual(result.score, 0.1)  # 0.0 + 0.1

    def test_wrong_and_bad_format(self):
        text = "随便写点什么 <answer>3</answer>"
        result = self.reward.score(text, "2")
        self.assertFalse(result.correct)
        self.assertFalse(result.format_ok)
        self.assertAlmostEqual(result.score, -0.1)  # 0.0 - 0.1

    def test_no_answer_extracted_gets_min_score(self):
        text = "这段输出完全没有按格式来"
        result = self.reward.score(text, "2")
        self.assertFalse(result.correct)
        self.assertIsNone(result.extracted_answer)
        self.assertEqual(result.score, -1.0)

    def test_reward_hacking_style_copy_number_from_question_still_checked_against_gt(self):
        # 典型 reward hacking 现象之一：模型复制题目中的数字凑答案，
        # 只要凑的数字和 ground_truth 不等价，规则reward应判定为错误。
        text = "<think>直接抄题目里的数字</think><answer>100</answer>"
        result = self.reward.score(text, "42")
        self.assertFalse(result.correct)


class TestComputeScoreInterface(unittest.TestCase):
    """验证 compute_score 与 verl.utils.reward_score.default_compute_score 签名兼容。"""

    def test_basic_call(self):
        score = compute_score(
            data_source="gsm8k",
            solution_str="<think>1+1=2</think><answer>2</answer>",
            ground_truth="2",
            extra_info={"split": "train", "index": 0},
        )
        self.assertIsInstance(score, float)
        self.assertGreater(score, 0)

    def test_accepts_extra_kwargs_without_error(self):
        # veRL 内部调用时可能传入额外 kwargs（如 sandbox_fusion_url 等），
        # compute_score 需要能够容忍多余参数而不报错。
        score = compute_score(
            data_source="math",
            solution_str="<think>...</think><answer>\\boxed{2}</answer>",
            ground_truth="2",
            extra_info=None,
            sandbox_fusion_url=None,
            concurrent_semaphore=None,
        )
        self.assertIsInstance(score, float)


if __name__ == "__main__":
    unittest.main(verbosity=2)
