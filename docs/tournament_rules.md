# Tournament Rules

These rules cover Team Deathmatch on the Official Leaderboard. We publish a
new leaderboard at 00:00 UK time on January 1, April 1, July 1 and October 1,
and the entry deadline is 72 hours before each. Your entry plays the nine
systems on the leaderboard and takes the lowest one's place if its Elo is
higher.

1. **Any JAX System.** Any system that runs as pure JAX and exports to our
   [portable model file](tournaments.md#how-we-run-and-score) may enter, sent as
   one Git commit that passes `entry check`.
2. **One Run, One Budget.** We train your entry once on
   [our RTX 5090](tournaments.md#how-we-run-and-score) with no internet for at
   most 10 hours, and it must stop itself within 255 million game steps,
   counting each tick of every game.
3. **Only Our Run Counts.** Your entry is the last model our run chose to
   enter, or else the last model it saved.
4. **Stay Off the Test Maps.** Your run trains and chooses its entry only on
   maps 0 to 46, never on test maps 47 to 51 or the built-in scenarios.
5. **Fast Enough to Play.** Your entry must play at least 5,000 game steps per
   second within 10 GB of GPU memory in `entry check`'s speed test on our card,
   which runs 512 games at once.
6. **Only What Your Team Sees.** Every action uses only what your team can
   see. Your program may act for each agent or for the whole team, and
   teammates may send each other messages. Whole-team programs and teammate
   messages open on December 10, 2026.
7. **Learn Only in Our Run.** Your run may bring nothing trained or recorded
   on MARL-BGs except released MARL-BGs models: training with or against them
   is free, but starting from, containing or imitating one adds its training
   steps to your 255 million.
8. **No Duplicates.** Submit as often as you like, but we skip entries that
   differ only in settings, such as seeds, hyperparameters, budget or reward
   weights, from a leaderboard system or an entry sent this season.
9. **Everything Is Open Source.** Your run saves a portable model at least
   every 5 million training steps, and we release your code, results and those
   models under Apache-2.0, entered or not, so everything you send must allow
   that.
10. **Play Fair.** Getting around a rule's purpose breaks it, and a break found
    later removes the entry and bars the account that sent it for one year.

How we run and score entries is in the
[tournaments guide](tournaments.md#enter-the-leaderboard). Ask us if a rule is
unclear.

## Execution and Teammate Messages

The leaderboard's What Each Entry Cost table shows how each entry runs, in two
columns.

**Execution.** Decentralized: the program decides for each agent separately,
using only that agent's view (which includes what its teammates see), its own
memory and the messages sent to it. Centralized: one program picks all five
actions together, so on the same tick it can also use each teammate's memory and
the actions it picks for them.

**Teammate Messages.** None: the program sends no messages. Next tick: the
program sends messages, and each one reaches its teammates one tick after it is
sent. A centralized entry shows a dash.

A centralized program's cost can grow with team size. Every leaderboard game is
5 v 5, so Play Speed and Play Memory show that cost at five agents only.
