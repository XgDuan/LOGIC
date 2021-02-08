# MNIST for Raven’s Progressive Matrices

## Introduction

In this paper, we considering constructing a toy dataset for the task of RPM. We will only consider
a very limit of rules and attributes.

Besides, we considering that the model should be training and testing under different rule distribu-
tion:
+ As in other datasets, the model should be trained with the same rule-attribute set and the same
    data distribution.
+ The model should be trained with part of the rules while testing with another part.
+ The model should be trained and tested under different rule distribution.

## The considered rules and attributes:
+ Rules: we consider the rules as in the `RAVEN', we consider four rules:
    + Const
    + Progression (+ / -)
    + Arithemetic (c = a + b)
    + Dist three (three component has different rules)
+ Attributes:
    + Number (The number of the given MNIST digits)
    + Rotation (We considering rotate the numbner 0, 90, 180, 280 digrees.)
        + Some number has trouble with rotation.
    + Color (We colorize these numbers)

## Dataset Generation Process


## Recent works:
[] Finish the dataset
[] CNN Baseline
[] Our models
    [] Basic Structure: Considering the optimizing problem
    [] Adcance Structure: Rules Shifting?