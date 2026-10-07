# Tournament Rulebook

**Version 1.** These are the official rules of the MARL-BattleGrounds
tournament for Team Deathmatch, from Season 1 onward. King of the Hills and
Capture the Flag get their own ladders and numbers later. How games become
ratings is defined in the
[tournament protocol](protocol.md#big-nine-tournament-and-baseline-library);
this rulebook sets who may enter, what a run may use, and the limits.

## Words Used Here

- **Game step:** one tick of one game. In 5v5, one game step holds 10 agent
  decisions.
- **System:** anything that returns a team's actions from its agents'
  permitted inputs.
- **Field:** the nine systems in a season's table, the Big Nine.
- **Member:** a system in the field. **Challenger:** a submitted system.
- **Maintainers:** the people who run MARL-BattleGrounds.

## The Season

- **One table a month.** A new table is published at 00:00 UK time
  (Europe/London) on the first day of each month and stays the same all
  month. Each season gives the exact instant in UTC and in your local time.
- **Cutoff: 72 hours before.** Submissions for a season close 72 elapsed
  hours before its table is published.
- **The Big Nine.** Every season's field holds nine systems.
- **Season 0** is our six baselines and three scripted bots, rated on the
  test maps. **Season 1 begins January 1, 2027.** Season 2 is February 2027,
  and so on.

## Rule 1. Who May Enter

1.1 **Any JAX System may enter:** learned, scripted, search-based, hybrid or
built on a language model. It must run as pure JAX and export to our
portable file.

1.2 The table shows each entry's type (Learned, Scripted, Hybrid or Language
Model) and whether it uses an outside pretrained model.

1.3 Season 0's three scripted bots are reference opponents built as part of
the benchmark. Rules 2 and 3 do not apply to them.

## Rule 2. Budget And Machine

2.1 **At most 255 million game steps and 8 hours.** Every game your run plays
counts, validation included. We stop the run at whichever limit comes first.

2.2 **The machine:** one RTX 5090 (32 GB), an AMD Ryzen 9 9950X3D (16 cores,
32 threads), 48 GB of RAM and 50 GB of disk. Your code runs sealed off: no
network, and it sees only its own folder.

2.3 Installing happens before the clock, without the GPU, in at most 1 hour.
List every download with its hash.

## Rule 3. What Your Run May Use

3.1 **Released MARL-BGs models** are free to use as training opponents or
teammates. Used as starting weights, as a frozen part of your entry, or as a
teacher you learn from, their training steps count toward your 255 million.

3.2 **Outside pretrained models** (for example a language model) are allowed
if they were public at least 90 days before the cutoff, were not made by
your team, never trained on MARL-BGs or a copy of it, and are licensed so we
can release your entry.

3.3 **Nothing else learned from MARL-BGs outside our run:** no weights,
tables, datasets or recorded games. Hand-chosen settings are fine.

3.4 **Maps 0-46 are yours** to use as you like. Your run never touches test
maps 47-51 or the scenarios. At home you may test a finished system on maps
47-51, but never choose anything by those results.

## Rule 4. Play Limits

Measured with our public check command: your system plays itself, 512 games
at once, on one RTX 5090. This is playing speed with the game running, not
training speed. We also check it in its official games.

4.1 **At least 5,000 game steps per second.**

4.2 **At most 10 GB of GPU memory.**

## Rule 5. Information

5.1 **Teammates may message each other, one tick late.** Within a tick, each
actor decides alone. A message sent on one tick arrives on the next.

5.2 **Adapt inside a game, not across games.** Each game stands alone: memory
starts fresh every game, and weights never change during evaluation.

5.3 **Shared sight.** Each agent sees what lies in its own sight and line of
sight, plus what its living teammates see (SharedObs). No system ever reads
hidden game data.

## Rule 6. Submitting

6.1 **One submission per team per season.** A team is its people and their
labs.

6.2 **One seat per team.** Your new system replaces your current member only
if it rates higher. The Season 0 systems are exempt. The maintainers count as
one team.

6.3 **Send a Git repository at one commit** with: a train command that keeps
its chosen entry saved in its output folder; a lock file; a load function
that only loads that entry, in at most 10 minutes; your config, training
maps, entry type and any starting models with their hashes.

6.4 Send it through the GitHub issue form.

6.5 We first run it for 15 minutes as a test. If the test fails, you may fix
it once without losing your submission or your place in the queue.

6.6 **Everything is open-sourced** under Apache-2.0, the license MARL-BGs
uses: code, results and all checkpoints, admitted or not, published with the
season's table.

6.7 **Every entrant is released** into one general pool you can download,
whether or not it joins the field. Each season's Big Nine also gets its own
folder. Leaving the field ends membership, not history.

## Rule 7. Our Run

7.1 **We run your script once,** on our internal RTX 5090 with nothing else
on it. That run is your entry: when it ends or hits a limit, we enter what it
last saved as its choice.

7.2 If your code fails, that uses up your season's submission. If our side
fails, we rerun it at no cost to you.

7.3 **Arrival order.** We check complete submissions one at a time, in the
order they arrive, against the field as it stands after the previous
challenger. The published table changes only when the next monthly table is
published.

7.4 **Unfinished work rolls forward** to the next season and keeps its place
in the queue.

7.5 On a GPU, our run does not repeat your home run exactly, so a lucky seed
may not carry over. Your script may turn on XLA's deterministic mode, which
repeats training exactly but trains much slower.

## Rule 8. The Evaluation

8.1 You are Team A. Spawn ends swap between paired games. 200 games against
each member, 40 on each test map 47-51, mirrored 5v5. First to 20 points
wins; at tick 300 it is a draw. Red Zone depth 5.0; 128 games at once.

8.2 Games run in XLA's deterministic mode. We draw the seed at random when your
games are played, after your training run, and publish it with your results,
so anyone with an RTX 5090 and our software lock can replay them exactly.

8.3 **Mirrored rosters.** Both teams use the same roster in the same order:
the standard five-class 5v5 of Mage, Warrior, Hunter, Rogue and Priest.

8.4 **Full records, no invented results.** Full metrics are recorded from the
first tick of every challenger game. The field's own games carry full metrics
too; Season 0's come from a replay that matched all 7,200 of its games. A
resource or write failure never becomes a made-up win, draw or loss.

8.5 **Scenarios do not count.** Scenario results never feed Elo.

## Rule 9. Ratings

9.1 One joint Elo fit of the field plus the challenger, average 1,200, refit
before each challenger, with 95% intervals. Elo is shown and compared as
whole numbers.

9.2 **A challenger enters if its Elo is higher than the lowest member's.**
Equal is not enough. The lowest member leaves.

9.3 If several members tie for last, the one with the lower expected score
against the field leaves; then the one admitted earliest (among Season 0
systems, the later one in Season 0's declared order).

9.4 Rank 1 of each season joins the Hall of Champions. A system that stays
Rank 1 gets one banner showing its reign.

9.5 Every challenger's result is published, admitted or not.

9.6 **Rank follows Elo.** Tiers are labels beside it: 200-Elo bands from
Iron (below 800) to Grandmaster (2,000 and up), each with five 40-Elo
divisions. Tiers never change rank, rating or admission.

9.7 A claim that a system moved tier or division must also give its Elo gain
with its interval.

9.8 Elo from different seasons cannot be compared: each season has its own
field.

9.9 **Local runs never count.** A run on your own computer never promotes
anything and never publishes a result.

## Rule 10. Fair Play

10.1 **A rule's purpose binds too.** Getting around a limit breaks it, even
if no line names the trick. Ask if unsure.

10.2 Our own new entries follow every rule, and we submit them before we open
outside submissions.

10.3 A rule break found later removes the entry, the member it pushed out
returns, and its people may not submit for 12 seasons.

10.4 Rules change only between seasons, announced at least 30 days before
the cutoff. Each table names its rulebook version.

## Why These Numbers

- **255 million game steps and 8 hours** sit just above our own slowest
  baseline. Our six baselines took about 3.6 to 7.0 hours to reach 250 million
  game steps on one RTX 5090, counting each first run and its continuation (see the
  [hyperparameter search record](../training/hyperparameter_search.md)), so
  every Season 0 learned entrant meets the limits a challenger faces.
- **The play limits** exist because every entrant becomes a downloadable
  training opponent for later challengers, so a slow or memory-hungry entrant
  would slow down everyone who trains against it. Every Season 0 entrant
  plays well above the speed floor and well under the memory limit.
- **One official run by us, on one fixed and published machine,** follows the
  MineRL 2020 and Procgen 2020 competitions, which also retrained every
  submission themselves under fixed limits on experience, time and hardware.
- **Teammate messages arrive one tick late,** as in classic learned
  communication research such as DIAL, where each agent acts on the messages
  sent on the previous step.
