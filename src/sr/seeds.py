"""Seed banks, ordered by distance from mathematics.

Every seed is a short idea written in plain words for this project (paraphrases,
not quotations), with the field it comes from. The banks are the experimental
variable: the question is whether the source field's distance from math matters.

  noise  random common words (controls for "any extra context and an extra step")
  polya  math problem-solving heuristics, after Polya's "How to Solve It" (same domain)
  near   principles from physics, engineering and computer science (near domain)
  far    ideas from philosophy, arts, nature, crafts and strategy (far domain)
"""
from __future__ import annotations

import random

POLYA = [
    ("math heuristics", "Ask: what is the unknown, what are the data, and what is the condition?"),
    ("math heuristics", "Recall a related problem you have already solved."),
    ("math heuristics", "Solve a simpler or special case first."),
    ("math heuristics", "Work backwards from what you want to find."),
    ("math heuristics", "Draw a figure and choose good notation."),
    ("math heuristics", "Introduce an auxiliary element that connects the data to the unknown."),
    ("math heuristics", "Generalize the problem; the general case can be easier."),
    ("math heuristics", "Split the problem into parts, solve them, and recombine."),
    ("math heuristics", "Make sure you have used all the data."),
    ("math heuristics", "Find a familiar problem that has the same unknown."),
    ("math heuristics", "Check the result, and try to derive it a second way."),
    ("math heuristics", "Try small cases and look for a pattern."),
    ("math heuristics", "Look for symmetry."),
    ("math heuristics", "Look for a quantity that never changes (an invariant)."),
    ("math heuristics", "Assume the opposite and look for a contradiction."),
    ("math heuristics", "Restate the problem in your own words."),
]

NEAR = [
    ("physics", "If something is symmetric, some quantity is conserved; find what stays the same."),
    ("physics", "Check the units; a correct answer must have the right dimensions."),
    ("physics", "Push the variables to extreme values, such as zero and infinity, and see what happens."),
    ("physics", "Nature often takes the path that makes some quantity smallest."),
    ("physics", "Choose the reference frame in which the motion looks simplest."),
    ("optics", "Reflect the picture in a mirror to turn a bent path into a straight line."),
    ("electrical engineering", "Move to another domain where the hard operation becomes easy, as convolution becomes multiplication."),
    ("electrical engineering", "Superposition: split the input into simple parts, solve each, and add the results."),
    ("control engineering", "Use feedback: compare the result with the goal and correct the error."),
    ("signal processing", "Sample enough points to see the pattern before you generalize."),
    ("linear algebra", "Change the basis so that the problem becomes diagonal and decoupled."),
    ("computer science", "Divide and conquer: split in halves, solve each, then combine."),
    ("computer science", "Store answers to small subproblems and build the big answer from them."),
    ("computer science", "Model the objects as nodes and the relations between them as edges."),
    ("probability", "Count the complement; what you do not want may be easier to count."),
    ("chemistry", "Balance both sides: count what goes in and what comes out."),
    ("thermodynamics", "Look for the balance point where two opposing tendencies cancel."),
    ("statistics", "Make a rough estimate first, then refine it."),
    ("software testing", "Write a test with small numbers before writing the general solution."),
    ("engineering design", "Build a quick prototype, test it, and then improve it."),
    ("numerical methods", "Start from a guess and improve it step by step until it stops changing."),
    ("information theory", "Ask which piece of information removes the most uncertainty."),
    ("mechanics", "Draw every force acting on the object, then add them."),
    ("geometry of physics", "Project the problem onto a simpler direction where it becomes one-dimensional."),
]

FAR = [
    ("Daoism", "Water wins against stone not by force but by flowing around it, patiently."),
    ("Daoism", "The usefulness of a bowl lies in the empty space inside it."),
    ("Daoism", "To make something strong, first see where it is weak."),
    ("Zen", "Before you change the picture, change where you are standing."),
    ("Stoicism", "Separate what is in your control from what is not, and spend effort only on the first."),
    ("military strategy", "Attack where the opponent is unprepared and avoid where they are strongest."),
    ("military strategy", "Arrange things so the outcome is decided before the battle begins."),
    ("evolution", "Many small variations, kept only when they survive, can build something complex."),
    ("ecology", "A burned forest regrows from its edges inward."),
    ("music", "A theme becomes clear when you hear it repeated with small variations."),
    ("music", "The silence between notes is part of the music."),
    ("painting", "Squint at the scene: the big shapes of light and dark matter before the details."),
    ("cooking", "Lay out every ingredient before you start to cook."),
    ("cooking", "Taste as you go, and adjust in small steps."),
    ("architecture", "Follow the path the weight takes down to the ground."),
    ("gardening", "Prune the branches that take strength without giving fruit."),
    ("navigation", "When you are lost, go back to the last place you were sure of."),
    ("martial arts", "Use the opponent's own force; redirect it instead of blocking it."),
    ("mountaineering", "Choose the route from the summit down, then climb it from the bottom up."),
    ("storytelling", "Every character wants something; find what each one wants."),
    ("medicine", "Diagnose before you treat."),
    ("weaving", "Strong cloth comes from threads that cross each other."),
    ("pottery", "Center the clay first; nothing good comes from an off-center start."),
    ("detective fiction", "Rule out the impossible; what remains must be true, however strange."),
    ("carpentry", "Measure twice, cut once."),
    ("translation", "Say the same thing in another language and its hidden structure shows."),
    ("theatre", "Change the stage and the same actors behave differently."),
    ("tracking animals", "Read the tracks to learn where the animal went, not where it is now."),
]

WORDS = """apple river window candle mirror ladder pocket cloud anchor violin pebble garden
lantern basket feather compass harbor meadow saddle teapot blanket chimney orchard
kettle marble paddle quilt ribbon sparrow tunnel umbrella velvet wagon yarn zipper
bridge castle desert forest island jungle valley glacier canyon volcano lighthouse
pencil eraser notebook stapler scissors bucket shovel hammer wrench bottle jacket
helmet slipper button necklace bracelet carpet curtain pillow mattress drawer
cabinet shelf barrel crate rope chain hook needle thread button cotton wool silk
salt pepper honey butter bread cheese onion garlic lemon cherry peach plum olive
tiger rabbit turtle dolphin falcon beetle spider walrus camel donkey goose otter""".split()
WORDS = list(dict.fromkeys(WORDS))  # drop repeats

BANKS = {"polya": POLYA, "near": NEAR, "far": FAR}


def noise_seed(rng: random.Random, n_words: int = 6) -> tuple[str, str]:
    return ("a list of random words", ", ".join(rng.sample(WORDS, n_words)))


def draw_seed(bank: str, rng: random.Random) -> tuple[str, str]:
    """Return (domain, seed_text)."""
    if bank == "noise":
        return noise_seed(rng)
    return rng.choice(BANKS[bank])
