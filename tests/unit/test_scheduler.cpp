#include <gtest/gtest.h>
#include "scheduler.h"
#include "test_helpers.h"
#include "temp_dir.h"
#include <ctime>

// =============================================================================
// Cron parsing tests
// =============================================================================

TEST(SchedulerTest, ParseCronEveryMinute) {
    Scheduler::CronExpr expr;

    EXPECT_TRUE(Scheduler::parse_cron("* * * * *", expr));
    EXPECT_EQ(expr.minutes.size(), 60u);
    EXPECT_EQ(expr.hours.size(), 24u);
    EXPECT_EQ(expr.days.size(), 31u);
    EXPECT_EQ(expr.months.size(), 12u);
    EXPECT_EQ(expr.weekdays.size(), 7u);
}

TEST(SchedulerTest, ParseCronSpecificValues) {
    Scheduler::CronExpr expr;

    EXPECT_TRUE(Scheduler::parse_cron("30 9 * * 1", expr));

    // Minute should only have 30
    EXPECT_EQ(expr.minutes.size(), 1u);
    EXPECT_EQ(expr.minutes.count(30), 1u);

    // Hour should only have 9
    EXPECT_EQ(expr.hours.size(), 1u);
    EXPECT_EQ(expr.hours.count(9), 1u);

    // Weekday should only have 1 (Monday)
    EXPECT_EQ(expr.weekdays.size(), 1u);
    EXPECT_EQ(expr.weekdays.count(1), 1u);
}

TEST(SchedulerTest, ParseCronRange) {
    Scheduler::CronExpr expr;

    EXPECT_TRUE(Scheduler::parse_cron("0 9-17 * * *", expr));

    // Hours 9-17 (inclusive) = 9 hours
    EXPECT_EQ(expr.hours.size(), 9u);
    EXPECT_EQ(expr.hours.count(9), 1u);
    EXPECT_EQ(expr.hours.count(17), 1u);
    EXPECT_EQ(expr.hours.count(8), 0u);
    EXPECT_EQ(expr.hours.count(18), 0u);
}

TEST(SchedulerTest, ParseCronStep) {
    Scheduler::CronExpr expr;

    EXPECT_TRUE(Scheduler::parse_cron("*/15 * * * *", expr));

    // Every 15 minutes: 0, 15, 30, 45
    EXPECT_EQ(expr.minutes.size(), 4u);
    EXPECT_EQ(expr.minutes.count(0), 1u);
    EXPECT_EQ(expr.minutes.count(15), 1u);
    EXPECT_EQ(expr.minutes.count(30), 1u);
    EXPECT_EQ(expr.minutes.count(45), 1u);
}

TEST(SchedulerTest, ParseCronStepEvery5) {
    Scheduler::CronExpr expr;

    EXPECT_TRUE(Scheduler::parse_cron("*/5 * * * *", expr));

    // Every 5 minutes: 0, 5, 10, ... 55
    EXPECT_EQ(expr.minutes.size(), 12u);
    EXPECT_EQ(expr.minutes.count(0), 1u);
    EXPECT_EQ(expr.minutes.count(5), 1u);
    EXPECT_EQ(expr.minutes.count(55), 1u);
}

TEST(SchedulerTest, ParseCronCommaList) {
    Scheduler::CronExpr expr;

    EXPECT_TRUE(Scheduler::parse_cron("0 9,12,18 * * *", expr));

    EXPECT_EQ(expr.hours.size(), 3u);
    EXPECT_EQ(expr.hours.count(9), 1u);
    EXPECT_EQ(expr.hours.count(12), 1u);
    EXPECT_EQ(expr.hours.count(18), 1u);
}

TEST(SchedulerTest, ParseCronRangeWithStep) {
    Scheduler::CronExpr expr;

    EXPECT_TRUE(Scheduler::parse_cron("0-30/10 * * * *", expr));

    // 0-30 with step 10: 0, 10, 20, 30
    EXPECT_EQ(expr.minutes.size(), 4u);
    EXPECT_EQ(expr.minutes.count(0), 1u);
    EXPECT_EQ(expr.minutes.count(10), 1u);
    EXPECT_EQ(expr.minutes.count(20), 1u);
    EXPECT_EQ(expr.minutes.count(30), 1u);
}

TEST(SchedulerTest, ParseCronWeekdayRange) {
    Scheduler::CronExpr expr;

    // Monday-Friday
    EXPECT_TRUE(Scheduler::parse_cron("0 9 * * 1-5", expr));

    EXPECT_EQ(expr.weekdays.size(), 5u);
    EXPECT_EQ(expr.weekdays.count(1), 1u);  // Monday
    EXPECT_EQ(expr.weekdays.count(5), 1u);  // Friday
    EXPECT_EQ(expr.weekdays.count(0), 0u);  // Sunday
    EXPECT_EQ(expr.weekdays.count(6), 0u);  // Saturday
}

// =============================================================================
// Cron validation tests
// =============================================================================

TEST(SchedulerTest, ValidateCronValid) {
    std::string error;

    EXPECT_TRUE(Scheduler::validate_cron("* * * * *", error));
    EXPECT_TRUE(Scheduler::validate_cron("0 9 * * 1-5", error));
    EXPECT_TRUE(Scheduler::validate_cron("*/5 * * * *", error));
    EXPECT_TRUE(Scheduler::validate_cron("0 9,12,18 * * *", error));
}

TEST(SchedulerTest, ValidateCronTooFewFields) {
    std::string error;
    EXPECT_FALSE(Scheduler::validate_cron("* * * *", error));
    EXPECT_FALSE(error.empty());
}

TEST(SchedulerTest, ValidateCronTooManyFields) {
    std::string error;
    EXPECT_FALSE(Scheduler::validate_cron("* * * * * *", error));
}

TEST(SchedulerTest, ValidateCronInvalidMinute) {
    std::string error;
    EXPECT_FALSE(Scheduler::validate_cron("60 * * * *", error));  // 60 > 59
}

TEST(SchedulerTest, ValidateCronInvalidHour) {
    std::string error;
    EXPECT_FALSE(Scheduler::validate_cron("* 25 * * *", error));  // 25 > 23
}

TEST(SchedulerTest, ValidateCronInvalidDay) {
    std::string error;
    EXPECT_FALSE(Scheduler::validate_cron("* * 32 * *", error));  // 32 > 31
}

TEST(SchedulerTest, ValidateCronInvalidMonth) {
    std::string error;
    EXPECT_FALSE(Scheduler::validate_cron("* * * 13 *", error));  // 13 > 12
}

TEST(SchedulerTest, ValidateCronInvalidWeekday) {
    std::string error;
    EXPECT_FALSE(Scheduler::validate_cron("* * * * 7", error));  // 7 > 6
}

// =============================================================================
// Time matching tests
// =============================================================================

TEST(SchedulerTest, MatchesTimeExact) {
    Scheduler::CronExpr expr;
    Scheduler::parse_cron("30 9 15 6 1", expr);  // 9:30 on 15th June, Monday

    std::tm tm = {};
    tm.tm_min = 30;
    tm.tm_hour = 9;
    tm.tm_mday = 15;
    tm.tm_mon = 5;   // June (0-based)
    tm.tm_wday = 1;  // Monday

    EXPECT_TRUE(Scheduler::matches_time(expr, tm));
}

TEST(SchedulerTest, MatchesTimeWrongMinute) {
    Scheduler::CronExpr expr;
    Scheduler::parse_cron("30 9 * * *", expr);

    std::tm tm = {};
    tm.tm_min = 31;  // Wrong minute
    tm.tm_hour = 9;
    tm.tm_mday = 15;
    tm.tm_mon = 5;
    tm.tm_wday = 1;

    EXPECT_FALSE(Scheduler::matches_time(expr, tm));
}

TEST(SchedulerTest, MatchesTimeWrongHour) {
    Scheduler::CronExpr expr;
    Scheduler::parse_cron("30 9 * * *", expr);

    std::tm tm = {};
    tm.tm_min = 30;
    tm.tm_hour = 10;  // Wrong hour
    tm.tm_mday = 15;
    tm.tm_mon = 5;
    tm.tm_wday = 1;

    EXPECT_FALSE(Scheduler::matches_time(expr, tm));
}

TEST(SchedulerTest, MatchesTimeWrongWeekday) {
    Scheduler::CronExpr expr;
    Scheduler::parse_cron("0 9 * * 1", expr);  // Only Mondays

    std::tm tm = {};
    tm.tm_min = 0;
    tm.tm_hour = 9;
    tm.tm_mday = 15;
    tm.tm_mon = 5;
    tm.tm_wday = 2;  // Tuesday

    EXPECT_FALSE(Scheduler::matches_time(expr, tm));
}

TEST(SchedulerTest, MatchesTimeEveryMinute) {
    Scheduler::CronExpr expr;
    Scheduler::parse_cron("* * * * *", expr);

    // Should match any time
    std::tm tm = {};
    tm.tm_min = 37;
    tm.tm_hour = 14;
    tm.tm_mday = 22;
    tm.tm_mon = 8;
    tm.tm_wday = 4;

    EXPECT_TRUE(Scheduler::matches_time(expr, tm));
}

// =============================================================================
// ScheduleEntry JSON serialization tests
// =============================================================================

TEST(SchedulerTest, ScheduleEntryToJson) {
    Scheduler::ScheduleEntry entry;
    entry.id = "abc123";
    entry.name = "test_schedule";
    entry.cron = "0 9 * * *";
    entry.prompt = "Good morning!";
    entry.enabled = true;
    entry.last_run = "2024-12-01T09:00:00";
    entry.created = "2024-11-01T10:00:00";

    nlohmann::json j = entry.to_json();

    EXPECT_EQ(j["id"], "abc123");
    EXPECT_EQ(j["name"], "test_schedule");
    EXPECT_EQ(j["cron"], "0 9 * * *");
    EXPECT_EQ(j["prompt"], "Good morning!");
    EXPECT_EQ(j["enabled"], true);
    EXPECT_EQ(j["last_run"], "2024-12-01T09:00:00");
    EXPECT_EQ(j["created"], "2024-11-01T10:00:00");
}

TEST(SchedulerTest, ScheduleEntryFromJson) {
    nlohmann::json j = {
        {"id", "xyz789"},
        {"name", "evening_check"},
        {"cron", "0 18 * * *"},
        {"prompt", "Check status"},
        {"enabled", false},
        {"last_run", "2024-12-02T18:00:00"},
        {"created", "2024-11-15T14:30:00"}
    };

    Scheduler::ScheduleEntry entry = Scheduler::ScheduleEntry::from_json(j);

    EXPECT_EQ(entry.id, "xyz789");
    EXPECT_EQ(entry.name, "evening_check");
    EXPECT_EQ(entry.cron, "0 18 * * *");
    EXPECT_EQ(entry.prompt, "Check status");
    EXPECT_FALSE(entry.enabled);
    EXPECT_EQ(entry.last_run, "2024-12-02T18:00:00");
    EXPECT_EQ(entry.created, "2024-11-15T14:30:00");
}

TEST(SchedulerTest, ScheduleEntryRoundtrip) {
    Scheduler::ScheduleEntry original;
    original.id = "test_id";
    original.name = "test_name";
    original.cron = "*/5 * * * *";
    original.prompt = "Test prompt";
    original.enabled = true;
    original.last_run = "";
    original.created = "2024-01-01T00:00:00";

    nlohmann::json j = original.to_json();
    Scheduler::ScheduleEntry loaded = Scheduler::ScheduleEntry::from_json(j);

    EXPECT_EQ(original.id, loaded.id);
    EXPECT_EQ(original.name, loaded.name);
    EXPECT_EQ(original.cron, loaded.cron);
    EXPECT_EQ(original.prompt, loaded.prompt);
    EXPECT_EQ(original.enabled, loaded.enabled);
    EXPECT_EQ(original.created, loaded.created);
}

// =============================================================================
// Next run time tests
// =============================================================================

TEST(SchedulerTest, FormatNextRun) {
    // This tests that format_next_run returns a non-empty string
    std::string next = Scheduler::format_next_run("* * * * *");
    EXPECT_FALSE(next.empty());
}

TEST(SchedulerTest, FormatNextRunInvalidCron) {
    std::string next = Scheduler::format_next_run("invalid");
    // Should return error or empty string
    EXPECT_TRUE(next.find("error") != std::string::npos || next.empty() ||
                next.find("Invalid") != std::string::npos);
}

// =============================================================================
// handle_sched_args tests
// =============================================================================

TEST(SchedulerTest, HandleSchedArgsHelp) {
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    int result = handle_sched_args({"help"}, callback);
    EXPECT_EQ(result, 0);
    EXPECT_TRUE(output.find("Usage") != std::string::npos ||
                output.find("sched") != std::string::npos);
}

TEST(SchedulerTest, HandleSchedArgsListEmpty) {
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    // List on a fresh scheduler might be empty or show headers
    int result = handle_sched_args({"list"}, callback);
    // Should succeed even with no schedules
    EXPECT_EQ(result, 0);
}

TEST(SchedulerTest, HandleSchedArgsInvalidCron) {
    std::string output;
    auto callback = [&output](const std::string& s) { output += s; };

    // Invalid cron expression
    int result = handle_sched_args({"add", "test", "invalid_cron", "prompt"}, callback);
    EXPECT_EQ(result, 1);
}

// =============================================================================
// Scheduler CRUD tests (require temp config path)
// =============================================================================

class SchedulerCRUDTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Note: Scheduler uses its own config path
        // These tests verify the API but may use the real config path
    }
};

TEST_F(SchedulerCRUDTest, AddAndGet) {
    Scheduler scheduler;

    std::string id = scheduler.add("test_schedule", "0 9 * * *", "Good morning!");
    EXPECT_FALSE(id.empty());

    Scheduler::ScheduleEntry* entry = scheduler.get(id);
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->name, "test_schedule");
    EXPECT_EQ(entry->cron, "0 9 * * *");
    EXPECT_EQ(entry->prompt, "Good morning!");
    EXPECT_TRUE(entry->enabled);

    // Clean up
    scheduler.remove(id);
}

TEST_F(SchedulerCRUDTest, GetByName) {
    Scheduler scheduler;

    std::string id = scheduler.add("find_me", "*/5 * * * *", "Test");

    // Find by name
    Scheduler::ScheduleEntry* entry = scheduler.get("find_me");
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(entry->id, id);

    // Clean up
    scheduler.remove(id);
}

TEST_F(SchedulerCRUDTest, EnableDisable) {
    Scheduler scheduler;

    std::string id = scheduler.add("toggle_test", "0 0 * * *", "Test");

    // Should be enabled by default
    EXPECT_TRUE(scheduler.get(id)->enabled);

    // Disable
    EXPECT_TRUE(scheduler.disable(id));
    EXPECT_FALSE(scheduler.get(id)->enabled);

    // Enable
    EXPECT_TRUE(scheduler.enable(id));
    EXPECT_TRUE(scheduler.get(id)->enabled);

    // Clean up
    scheduler.remove(id);
}

TEST_F(SchedulerCRUDTest, Remove) {
    Scheduler scheduler;

    std::string id = scheduler.add("remove_me", "0 0 * * *", "Test");
    EXPECT_NE(scheduler.get(id), nullptr);

    EXPECT_TRUE(scheduler.remove(id));
    EXPECT_EQ(scheduler.get(id), nullptr);
}

TEST_F(SchedulerCRUDTest, RemoveNonexistent) {
    Scheduler scheduler;
    EXPECT_FALSE(scheduler.remove("nonexistent_id"));
}

TEST_F(SchedulerCRUDTest, List) {
    Scheduler scheduler;

    // Add some schedules
    std::string id1 = scheduler.add("sched1", "0 9 * * *", "Morning");
    std::string id2 = scheduler.add("sched2", "0 18 * * *", "Evening");

    std::vector<Scheduler::ScheduleEntry> list = scheduler.list();
    EXPECT_GE(list.size(), 2u);

    // Clean up
    scheduler.remove(id1);
    scheduler.remove(id2);
}
