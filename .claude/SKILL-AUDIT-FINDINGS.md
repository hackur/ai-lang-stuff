# Claude Code Skills Audit - Incorrect Assumptions

**Date**: October 2025
**Status**: In Progress
**Purpose**: Document all incorrect assumptions in the initial skills to ensure corrected versions are accurate

---

## ❌ CRITICAL ERRORS FOUND

### 1. Constants vs Enums Terminology

**WRONG** ❌:
- Referred to constants as "enums"
- Suggested PHP 8.1+ enum syntax
- Implied they use backed enums

**ACTUAL** ✅:
- PCR Card uses **class constants** (not PHP enums)
- Pattern: `public const NAME = 'value';`
- Only ONE class in `app/Enums/` (UserStatus) and it's still a class with constants
- NO PHP 8.1+ enum syntax used anywhere

**Impact**: Terminology throughout all skills is incorrect

---

### 2. State Constants Architecture

**WRONG** ❌:
```php
// I described states as:
const DRAFT = 'draft';           // String values
const SUBMITTED = 'submitted';
```

**ACTUAL** ✅:
```php
// States store CLASS NAMES not strings:
public const DRAFT = \App\States\Draft::class;
public const SUBMITTED = \App\States\Submitted::class;
```

**Key Insight**: PCR Card uses a **dual-class architecture**:
1. `App\Constants\SubmissionState` - Type-safe constants (99% of usage)
2. `App\States\SubmissionState` - Spatie state machine base class (only for $casts and instanceof)

**Impact**: state-machine-handler skill has fundamental misunderstanding of architecture

---

### 3. Constants Pattern and Methods

**WRONG** ❌:
- Suggested constants with basic values only
- Didn't document standard helper methods

**ACTUAL** ✅:
ALL constants classes follow this pattern:
```php
class PromoCodeType
{
    public const TYPE_FIXED = 'fixed';
    public const TYPE_PERCENTAGE = 'percentage';

    // Standard methods ALL constants have:
    public static function all(): array;          // Returns all values
    public static function label(string $type): string;  // Human-readable label
    public static function options(): array;      // For Select fields: ['value' => 'Label']
    public static function isValid(string $type): bool;  // Validation helper
}
```

**Additional State-Specific Methods**:
```php
class SubmissionState
{
    // All the above PLUS:
    public static function terminalStates(): array;      // End states
    public static function activeStates(): array;        // Workflow states
    public static function displayLabels(): array;       // Nova display helpers
    public static function loadingLabels(): array;       // Nova badge styling
    public static function failedLabels(): array;        // Nova badge styling
    public static function successLabels(): array;       // Nova badge styling
}
```

**Impact**: Constants usage examples in all skills need updating

---

### 4. Constants Naming Convention

**WRONG** ❌:
- Used inconsistent naming (some with prefixes, some without)

**ACTUAL** ✅:
Constants use **prefixed naming** for clarity:
- `PromoCodeType::TYPE_FIXED` (not `PromoCodeType::FIXED`)
- `ManualPaymentMethod::METHOD_CASH` (not `ManualPaymentMethod::CASH`)
- `ManualPaymentStatus::STATUS_PENDING` (not `ManualPaymentStatus::PENDING`)

**Exception**: State constants don't use prefixes:
- `SubmissionState::DRAFT` (not `SubmissionState::STATE_DRAFT`)
- `CardState::RECEIVED` (not `CardState::CARD_RECEIVED`)

**Impact**: All code examples showing constants need correction

---

### 5. Nova Badge Field Pattern

**WRONG** ❌:
```php
// I described this pattern:
Badge::make('Status')
    ->map(fn($value) => [
        'draft' => NovaBadgeType::SUCCESS,  // Using constants
        'submitted' => NovaBadgeType::INFO,
    ][$value] ?? NovaBadgeType::WARNING);
```

**ACTUAL** ✅:
```php
// Actual pattern uses HARDCODED STRINGS:
Badge::make('Usage Status', function () {
    return 'available';  // Return a value from closure
})
->map([
    'available' => 'success',   // Hardcoded string badge types
    'limited' => 'warning',
    'exhausted' => 'danger',
])
->label(function ($value) {
    return match ($value) {
        'available' => 'Available',
        'limited' => 'Limited',
        'exhausted' => 'Exhausted',
    };
});
```

**Key Differences**:
1. Badge field takes a **closure** that returns a calculated value
2. `.map()` uses **hardcoded strings** ('success', 'warning', 'danger', 'info')
3. `.label()` provides **display text** for each value
4. **NovaBadgeType constants are NOT used in Nova resources**

**Impact**: nova-resource-builder skill has completely wrong Badge pattern

---

### 6. Select Fields with Constants

**WRONG** ❌:
```php
// I suggested passing constants directly:
Select::make('Type')
    ->options([
        PromoCodeType::TYPE_FIXED => 'Fixed',
        PromoCodeType::TYPE_PERCENTAGE => 'Percentage',
    ]);
```

**ACTUAL** ✅:
```php
// Use the options() method:
Select::make('Type')
    ->options(PromoCodeType::options())  // Method returns the array
    ->displayUsingLabels();
```

**Impact**: All Select field examples need correction

---

### 7. States Directory Structure

**WRONG** ❌:
```
app/States/
├── Submission/
│   ├── SubmissionState.php
│   ├── Draft.php
│   └── Submitted.php
└── Card/
    ├── CardState.php
    └── CardReceived.php
```

**ACTUAL** ✅:
```
app/States/
├── SubmissionState.php    # Base class for submission states
├── CardState.php          # Base class for card states
├── Draft.php              # Submission state
├── Submitted.php          # Submission state
├── Received.php           # Submission state
├── Completed.php          # Submission state
├── Shipped.php            # Submission state
├── Cancelled.php          # Submission state
├── CardReceived.php       # Card state
├── CardAssessment.php     # Card state
├── CardInProgress.php     # Card state
├── CardQualityCheck.php   # Card state
├── CardCompleted.php      # Card state
├── CardLabelSlab.php      # Card state
└── CardCancelled.php      # Card state
```

**Key**: States are in `app/States/` FLAT structure, not nested subdirectories

**Impact**: state-machine-handler skill has wrong file paths

---

## 📋 Complete Constants List (18 Classes)

From `app/Constants/`:
1. `PaymentStatus` - Payment processing statuses
2. `SubmissionState` - Submission state class references
3. `CardState` - Card state class references
4. `CardGrades` - Card grading values
5. `ManualPaymentMethod` - Offline payment methods
6. `ManualPaymentStatus` - Manual payment statuses
7. `NovaBadgeType` - Nova badge type strings (but NOT used in resources!)
8. `PromoCodeFilterStatus` - Nova filter status options
9. `PromoCodePurpose` - Promo code categorization
10. `PromoCodeType` - Discount type (fixed/percentage)
11. `QrCodeAction` - QR code action types
12. `QrCodeStatus` - QR code statuses
13. `Role` - User role names
14. `ServiceAddonApplication` - Service addon application levels
15. `ServiceCategory` - Service categorization
16. `ServicePricingType` - Service pricing types
17. `ServiceSlugs` - Service slug constants
18. `SubmissionPriceChangeStatus` - Price change request statuses

From `app/Enums/`:
1. `UserStatus` - User account statuses (ACTIVE, INACTIVE, PENDING)

**Note**: UserStatus is in `app/Enums/` but is still a class with constants, NOT a PHP 8.1+ enum

---

### 8. Service Layer Directory Structure

**WRONG** ❌:
```
app/Services/
├── Payment/
│   ├── PaymentManager.php
│   ├── FlexiblePricingService.php
│   └── PromoCodeService.php
└── State/
    ├── StateTransitionService.php
    └── StateTransitionValidator.php
```

**ACTUAL** ✅:
```
app/Services/
├── PaymentManager.php              # Flat structure (not in subdirectory)
├── FlexiblePricingService.php
├── PromoCodeService.php
├── StateTransitionService.php
├── StateTransitionValidator.php
├── PaymentProcessors/              # Processors in subdirectory
│   └── StripePaymentProcessor.php
├── SubmissionPriceChange/          # Module-specific subdirectory
│   ├── SubmissionPriceChangeService.php
│   └── Actions/
│       ├── ApproveSubmissionPriceChange.php
│       └── RejectSubmissionPriceChange.php
└── Development/                    # Development helpers
    ├── SubmissionBuilder.php
    └── StateProgressionHelper.php
```

**Key**: Services are mostly FLAT in `app/Services/`, NOT organized in subdirectories by domain

---

### 9. Service Return Patterns

**WRONG** ❌:
- Didn't document consistent return structure
- Examples showed raw returns or exceptions

**ACTUAL** ✅:
All services use **consistent array return structure**:
```php
return [
    'success' => true|false,    // or 'valid' => true|false
    'message' => 'Human-readable message',
    'data' => [...]             // Structured data
];
```

Example from PromoCodeService:
```php
public function applyPromoCode(string $code, Submission $submission, User $user): array
{
    // Validation...
    if ($error) {
        return [
            'success' => false,
            'message' => 'Promo code is invalid',
        ];
    }

    // Success...
    return [
        'success' => true,
        'message' => 'Promo code applied successfully',
        'data' => [
            'discount_amount' => $discountAmount,
            'final_amount' => $finalAmount,
        ],
    ];
}
```

---

### 10. Utility Scripts Missing from Skills

**WRONG** ❌:
- Skills showed raw artisan commands
- Didn't reference existing utility scripts

**ACTUAL** ✅:
PCR Card has 12 utility scripts in `scripts/`:
- `dev.sh` - Primary development workflow tool
- `staging.sh` - Staging deployment and management
- `prod.sh` - Production management (HUMAN-ONLY)
- `brand.sh` - Branding and logo generation
- `setup-websockets.sh` - WebSocket configuration
- And 7 more specialized scripts

**Key Development Commands**:
```bash
./scripts/dev.sh fresh              # Reset DB with all seeders
./scripts/dev.sh test               # Run PHPUnit tests
./scripts/dev.sh visible-test       # Run Dusk browser tests
./scripts/dev.sh seed:structure     # Roles & permissions only
./scripts/dev.sh clear:cache        # Clear all caches
./scripts/dev.sh up                 # Start containers
./scripts/dev.sh validate:routes    # Validate routes
./scripts/dev.sh validate:nova-search  # Validate Nova search
```

**Impact**: All skills should reference `./scripts/dev.sh` commands FIRST, then show underlying artisan commands

---

## ⏳ Still To Audit

- [ ] Database seeder actual patterns (Core/, ReferenceData/, Development/)
- [ ] Migration actual patterns and conventions
- [ ] Email template structure and theme usage
- [ ] API resource and controller patterns
- [ ] Dusk test configuration (DuskTestCase)
- [ ] PHPUnit/Pest test actual patterns
- [ ] Nova tab panel vs Panel usage (which resources use which?)
- [ ] Form request validation patterns
- [ ] Policy and authorization actual patterns

**Estimated Remaining**: 9 major areas to audit before skills can be corrected

---

## 🎯 Action Items for Skill Correction

### Content Corrections
1. **Remove all "enum" terminology** - Replace with "class constants"
2. **Fix state constant examples** - Use `::class` references, not strings
3. **Document constants pattern** - all(), label(), options(), isValid() methods
4. **Fix Badge field pattern** - Use closures + hardcoded string maps
5. **Fix Select field pattern** - Use `->options(ConstantClass::options())`
6. **Update directory structures** - States are flat, not nested
7. **Add constants naming conventions** - Prefixed naming (TYPE_, METHOD_, STATUS_)
8. **Document dual-class state architecture** - Constants vs State machine classes

### Make Skills More Concise
9. **Reference utility scripts FIRST** - Show `./scripts/dev.sh` commands before raw artisan
10. **Remove verbose examples** - Keep skills focused on patterns, not full implementations
11. **Link to existing docs** - Reference CLAUDE.md and docs/ instead of duplicating
12. **Focus on PCR Card specifics** - Don't duplicate Laravel/Nova standard docs

### Utility Script Reference Pattern

**BEFORE** (verbose):
```bash
# Run all tests
php artisan test

# Run specific test
php artisan test tests/Unit/PromoCodeTest.php

# Run with coverage
php artisan test --coverage
```

**AFTER** (concise):
```bash
# Use dev.sh wrapper
./scripts/dev.sh test                    # Run all tests
./scripts/dev.sh test:file <path>        # Run specific test

# See dev.sh help for all options
./scripts/dev.sh help
```

---

## 📝 Skill Structure Template (Concise Version)

Each skill should be:
- **Max 200-300 lines** (not 600+ lines)
- **Reference dev.sh first**, raw commands second
- **Focus on PCR Card patterns**, not Laravel basics
- **Link to docs**, don't duplicate them

**Example Structure**:
```markdown
# Skill Name

## When to Use
- Trigger 1
- Trigger 2

## Quick Reference (dev.sh commands)
./scripts/dev.sh command1
./scripts/dev.sh command2

## PCR Card Patterns (unique to this project)
- Pattern 1 with example
- Pattern 2 with example
- Pattern 3 with example

## Constants Used
- ConstantClass::CONSTANT_NAME

## Common Pitfalls
- Mistake 1 and correct approach
- Mistake 2 and correct approach

## Documentation Links
- CLAUDE.md: Section X
- docs/path/to/relevant-doc.md
```

---

**Next Steps**: Rebuild skills with concise, script-focused approach.
