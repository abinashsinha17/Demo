# Reconciliation Skill

This document defines the automated logic for the **Reconciliation Skill** agent within the NAV Exception Management ecosystem.

## Overview
The Reconciliation Skill is responsible for evaluating breaks between internal systems and external data providers (like custodians). It examines parameters such as:
- Break Type (e.g., Cash Break, Position Break)
- Variance (The absolute dollar difference)
- Confidence Score (Machine learning probability)
- Age of Escalation

## Skill Logic Implementation

The skill logic has been extracted into a standalone Python file (`reconciliation_skill.py`) for cleaner integration and easier testing.
It exposes a `ReconciliationSkill` class which processes these exceptions and routes them accordingly.
