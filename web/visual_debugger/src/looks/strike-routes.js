/**
 * @file The activation footprint table shared by the Basic and Ultimate looks
 * and the plan's collision allocator.
 *
 * `basics.js` (Basic activations) and `ultimates.js` (Ultimate activations)
 * each draw their own picture; this file now holds only ACTIVATION_FOOTPRINTS,
 * the CSS-pixel sizes the plan (`choreography-plan.js`) reserves for each
 * activation, so the allocator and the looks agree. The old shared route drawing
 * (route paths, direction arrows, the moving particle and the Charge "A → B"
 * label) is gone. Pure data: no DOM, no events, nothing read about any body.
 */

/**
 * Today's activation footprints in CSS pixels. Stroke-inclusive radii and flare
 * extents fit inside their footprint; route padding leaves out transparent hit
 * targets and underpainting. These numbers match the plan's allocator table.
 */
export const ACTIVATION_FOOTPRINTS = Object.freeze({
  activation: Object.freeze({
    basic_damage: Object.freeze({ width: 44, height: 44 }),
    basic_heal: Object.freeze({ width: 44, height: 44 }),
    holy_word: Object.freeze({ width: 56, height: 56, flareExtent: 27 }),
    hunter_trap: Object.freeze({ width: 27, height: 27, flareExtent: 25 }),
    rogue_poison: Object.freeze({ width: 48, height: 48 }),
    warrior_charge: Object.freeze({
      width: 32.16,
      height: 27.52,
      flareExtent: 26,
    }),
    mage_burst: Object.freeze({ width: 70, height: 70, flareExtent: 34 }),
    local: Object.freeze({ width: 48, height: 48 }),
  }),
  chargeOwnership: Object.freeze({ width: 72, height: 22, labelLength: 60 }),
  route: Object.freeze({
    // The standard activation envelope is the animated r3 particle plus its 3px
    // glow and 1px clearance. Holy Word alone needs 8px for its r4 particle and
    // 4px glow (and its 5px route stroke plus 5px glow spans 7.5px).
    activationPathPadding: Object.freeze({ default: 7, holy_word: 8 }),
    chargePathPadding: 1.5,
    activationMarkerPadding: 17,
    activationCompactMarkerPadding: 8,
    chargeMarkerPadding: 14,
  }),
});
