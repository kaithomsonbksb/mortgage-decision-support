import unittest

# test for low score -> no flags standard
# test for medium score -> enhanced review
# test for high score -> senior review

# low score + LTV 91% -> enhanced review
# low score + previous default -> enhanced review
# missing credit score -> enhanced review
# score of .4 -> senior review, 0.2 enhanced (boundary test)
# LTV 90 does not flag, 91 does
