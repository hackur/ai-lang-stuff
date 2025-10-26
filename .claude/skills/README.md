# PCR Card Claude Code Skills

Comprehensive collection of Claude Code skills for PCR Card development. These skills provide expert assistance for specific development tasks and automatically activate based on context.

## Available Skills

### Laravel Nova Administration

**[nova-resource-builder](nova-resource-builder/SKILL.md)**
- Creating and modifying Nova resources
- Configuring search arrays
- Implementing tab-based panel layouts
- Badge fields with closure pattern
- Nova 5.x best practices

**[laravel-package-specialist](laravel-package-specialist/SKILL.md)**
- Laravel and Nova package development
- Forked package management (pcrcard/nova-*)
- VCS path repositories and symlinks
- Webpack configuration for Nova packages
- Package versioning and git workflows

### Database Operations

**[database-seeder-builder](database-seeder-builder/SKILL.md)**
- Creating idempotent seeders
- Core namespace pattern
- Environment-specific seed data
- Roles, permissions, services

**[migration-builder](migration-builder/SKILL.md)**
- Database schema migrations
- Adding columns and indexes
- Foreign key constraints
- Schema-only changes (not data)

### Testing

**[dusk-test-builder](dusk-test-builder/SKILL.md)**
- Laravel Dusk browser tests
- Visible-test configuration (headed Chrome)
- UI interaction testing
- Nova admin testing

**[phpunit-test-builder](phpunit-test-builder/SKILL.md)**
- PHPUnit unit tests
- Pest PHP syntax
- Feature tests
- API testing
- Mocking and fakes

### Feature Implementation

**[state-machine-handler](state-machine-handler/SKILL.md)**
- Two-level state machine (Submission + Card states)
- State transitions with Spatie Model States
- Validation and permissions
- State-based field visibility

**[payment-promo-code-handler](payment-promo-code-handler/SKILL.md)**
- Payment processing (Stripe, manual)
- Promo code validation and application
- Pricing calculations
- Price change requests

### Infrastructure

**[email-template-builder](email-template-builder/SKILL.md)**
- Laravel markdown email templates
- PCR Card custom theme
- Email-safe CSS practices
- Testing with Mailpit

**[service-class-builder](service-class-builder/SKILL.md)**
- Service layer architecture
- Business logic separation
- Single responsibility pattern
- Dependency injection

**[api-endpoint-builder](api-endpoint-builder/SKILL.md)**
- RESTful API endpoints
- Laravel API Resources
- Sanctum authentication
- Form request validation

## How Skills Work

Skills are **model-invoked** - Claude automatically activates them based on your request and the skill's description. You don't need to manually invoke skills; they activate when relevant.

### Automatic Activation

Claude identifies when to use skills based on:
- Keywords in your request
- Context of the conversation
- Task requirements
- Trigger words in skill descriptions

### Manual Invocation

You can also explicitly request a skill:
```
"Use the nova-resource-builder skill to create a new resource"
"Help me with this using the state-machine-handler skill"
```

## Skill Organization

Skills are organized by functional area:

- **Administration**: Nova resources, configuration
- **Database**: Seeders, migrations, schema
- **Testing**: Dusk browser tests, PHPUnit unit/feature tests
- **Features**: State machine, payments, promo codes
- **Infrastructure**: Email, services, API

## Common Triggers

### Nova Administration
`nova resource`, `badge field`, `nova search`, `nova tabs`, `nova lens`, `nova filter`

### Package Development
`package`, `nova field`, `nova tool`, `webpack.mix.js`, `pcrcard/nova-*`, `packages/`, `forked package`

### Database
`seeder`, `migration`, `roles and permissions`, `database schema`, `add column`

### Testing
`dusk test`, `browser test`, `unit test`, `feature test`, `phpunit`, `visible test`

### Features
`state transition`, `workflow`, `payment`, `promo code`, `discount`, `stripe`

### Infrastructure
`email template`, `service class`, `api endpoint`, `business logic`

## Best Practices

### Using Skills Effectively

1. **Be Specific**: Include relevant keywords from skill triggers in your requests
2. **Provide Context**: Explain what you're trying to accomplish
3. **Trust Automation**: Let Claude choose the right skill based on context
4. **Combine Skills**: Complex tasks may activate multiple skills sequentially

### Example Requests

**Good**:
- "Create a Nova resource for ManualPayment with tab panels and badge fields"
- "Add a migration to add tracking_number column to submissions table"
- "Write a Dusk test to verify the state transition from draft to submitted"
- "Build an API endpoint for promo code validation with Sanctum auth"

**Less Effective**:
- "Help with Nova" (too vague)
- "Fix this" (no context)
- "Make it work" (unclear goal)

## Skill Updates

Skills are living documents that evolve with the codebase. When you notice:
- New patterns emerging
- Best practices changing
- Missing information

Update the relevant SKILL.md file to keep documentation current.

## Related Documentation

Each skill references relevant project documentation:
- `docs/` - Comprehensive project documentation
- `CLAUDE.md` - Development guide and conventions
- Laravel/Nova docs - Official framework documentation

## Getting Help

### For Skill Usage
- Read the skill's SKILL.md file
- Check the "When to Use This Skill" section
- Review example patterns and checklists

### For Development
- Consult `CLAUDE.md` for general guidelines
- Review `docs/` for architectural decisions
- Check Laravel/Nova documentation for framework features

---

**Last Updated**: October 2025
**Total Skills**: 11
**Coverage**: Nova admin, package development, database, testing, features, infrastructure
