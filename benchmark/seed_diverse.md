| tool | scenario | seed_utterance | number | expected_outcome | notes |
| --- | --- | --- | --- | --- | --- |
| create | create_success_exact_time | Remind me tomorrow at 8:00 AM to take my blood pressure medicine. | 100 | success | clear create intent; exact clock time + explicit task; rotate beyond medicine |
| create | create_success_relative_daypart | Please remind me this Friday evening to pay the electricity bill. | 100 | success | clear create intent; relative daypart/date phrase |
| create | create_success_routine_anchor | Remind me after lunch to take a short walk downstairs. | 80 | success | time anchored to daily routine rather than clock |
| create | create_success_short_command | Tomorrow 6 pm, remind me to bring in the laundry. | 80 | success | compressed command style with time + task |
| create | create_success_errand_payment | Set a reminder for Monday morning to renew my bus card. | 80 | success | errands, payment, community paperwork, phone top-up |
| create | create_success_sports_activity | Remind me at 4:30 this afternoon to play table tennis with Lao Chen. | 80 | success | sports and social activities: table tennis, tai chi, square dance, walking group |
| create | create_success_personal_care | Please remind me tomorrow afternoon to get a haircut. | 80 | success | personal care: haircut, foot bath appointment, pharmacy pickup, glasses repair |
| create | create_success_household_safety | Before I go to bed tonight, remind me to close the balcony window. | 80 | success | household safety and chores: stove, windows, trash, plants, pets |
| create | create_success_appointment | Remind me next Tuesday at 9 AM to go to the dentist. | 80 | success | appointment reminders: dentist, clinic, bank, community office |
| create | create_success_family_social | Could you remind me after dinner to video call my sister? | 80 | success | family/social contact is one task type, not its own benchmark ability |
| create | create_success_cooking_food | Remind me before dinner to put the soup in the fridge. | 80 | success | cooking and food-prep tasks |
| create | create_success_delivery_shopping | Please remind me this evening to pick up the parcel from downstairs. | 80 | success | delivery, grocery, parcel, shopping tasks |
| create | create_missing_time_task_clear | Remind me to buy vegetables on the way home. | 100 | missing_fields | missing time_text; task is clear |
| create | create_missing_task_time_clear | Remind me tomorrow at 3:00 PM. | 100 | missing_fields | missing task; time_text is clear |
| create | create_missing_both_minimal | I need a reminder. | 80 | missing_fields | missing both task and time_text |
| query | query_by_task_success | What time is my table tennis reminder tomorrow? | 80 | success | query by task/entity |
| query | query_by_time_success | What do I have scheduled tomorrow at 6 PM? | 80 | success | query by time slot |
| query | query_by_day_success | Can you check all my reminders for this Friday? | 80 | success | query by date/day range |
| query | query_vague_memory_success | I forgot what I need to do tomorrow, can you check my reminders? | 80 | success | fuzzy memory query |
| query | query_polite_elder_style | Could you help me check whether I have any reminder for tomorrow morning? | 60 | success | polite elder style query |
| query | query_not_found_sports | What time is my tennis reminder tomorrow? | 80 | not_found | unmatched sports/activity reminder |
| query | query_not_found_appointment | Do I have a reminder for a dentist visit this afternoon? | 80 | not_found | unmatched appointment reminder |
| query | query_not_found_errand | Did I set a reminder to return the library book today? | 60 | not_found | unmatched errand reminder |
| update | update_success_time_shift | Move my table tennis reminder from 4:30 to 5:00 this afternoon. | 80 | success | update an identifiable reminder time |
| update | update_success_task_change | Change my reminder tomorrow at 8 AM from walking to checking blood sugar. | 80 | success | update task content for an identifiable reminder |
| update | update_success_reschedule_with_context | I cannot go for the haircut in the morning; change that reminder to tomorrow afternoon. | 80 | success | reschedule with contextual locator |
| update | update_ambiguous_generic | Change my reminder tomorrow to 6 PM. | 80 | ambiguous | new value is clear but target reminder is ambiguous |
| update | update_ambiguous_task_only | Move my medicine reminder to tonight. | 60 | ambiguous | task-only locator may match multiple reminders |
| update | update_not_found_sports | Change my tennis reminder tomorrow to 6 PM. | 60 | not_found | no matching sports reminder |
| update | update_not_found_personal_care | Move my haircut reminder to Friday afternoon. | 60 | not_found | no matching personal-care reminder |
| update | update_missing_target | Please update my reminder to 6 PM. | 80 | missing_fields | missing locator/target reminder |
| delete | delete_success_explicit | I will not play table tennis tonight, cancel that reminder. | 80 | success | explicit deletion target with context |
| delete | delete_success_reason_context | My daughter already called me back, please remove the call reminder for tonight. | 100 | success | delete with real-world reason and identifiable target |
| delete | delete_ambiguous_generic | Cancel my reminder tomorrow. | 80 | ambiguous | ambiguous delete; multiple possible reminders |
| delete | delete_ambiguous_task_only | Delete my medicine reminder. | 60 | ambiguous | task-only delete may match multiple reminders |
| delete | delete_not_found_sports | I am not going to play tennis this afternoon, cancel that reminder. | 60 | not_found | no matching sports reminder |
| delete | delete_not_found_appointment | Cancel my dentist reminder for this afternoon. | 60 | not_found | no matching appointment reminder |
| delete | delete_missing_target | Please delete one reminder for me. | 80 | missing_fields | missing locator/target reminder |
| no_tool | no_tool_daily_chat | The weather is nice today, I just went for a walk in the park. | 80 | no_tool | casual chat; no reminder intent |
| no_tool | no_tool_emotional_support | I feel a bit lonely tonight and want someone to talk to. | 80 | no_tool | emotional support; no reminder operation |
| no_tool | no_tool_happy_share | My grandson visited me today, I am really happy. | 60 | no_tool | positive sharing; no reminder operation |
| no_tool | no_tool_time_words_no_intent | Tomorrow afternoon I might drink tea with friends, just saying. | 80 | no_tool | contains time words but no reminder request |
| no_tool | no_tool_question_general | Do you think walking every day is good for blood pressure? | 60 | no_tool | general wellness question; no tool |
| no_tool | no_tool_future_plan_no_request | I may get a haircut next week if the weather is good. | 80 | no_tool | future plan mention but no create/query/update/delete intent |
