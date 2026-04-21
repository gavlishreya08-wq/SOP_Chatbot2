import unittest

from backend.rag.preprocess import canonical_section_title, extract_sections


class PreprocessTests(unittest.TestCase):
    def test_numbered_responsibility_item_is_not_section_heading(self):
        self.assertIsNone(
            canonical_section_title("11) Follow standards and best practices and policies.")
        )

    def test_numbered_responsibility_item_stays_in_responsibilities_section(self):
        sections = extract_sections(
            "\n".join(
                [
                    "RESPONSIBILITIES",
                    "10) Any other Job/Role/ Responsibility at any location given by the management from time to time.",
                    "11) Follow standards and best practices and policies.",
                    "TRAINING REQUIREMENT",
                    "Yearly minimum 64 hours of training.",
                ]
            )
        )

        self.assertIn(
            (
                "Responsibilities",
                "10) Any other Job/Role/ Responsibility at any location given by the management from time to time.\n"
                "11) Follow standards and best practices and policies.",
            ),
            sections,
        )
        self.assertIn(
            ("Training Requirement", "Yearly minimum 64 hours of training."),
            sections,
        )


if __name__ == "__main__":
    unittest.main()
