# scripts/setup_benchmark_data.py

"""
setup_benchmark_data.py

Purpose
-------
Create sample benchmark datasets for the Requirement Assistant Platform.

This script creates:

1. requirement_embedding_pairs.jsonl
   - Level 1 benchmark
   - Positive = relevant requirement
   - Negative = unrelated requirement

2. requirement_chunks.jsonl
   - Requirement chunks used for ranking benchmark

3. retrieval_eval.jsonl
   - Level 2 benchmark
   - Expected correct chunk ID for each query

4. llm_artifact_eval.jsonl
   - Level 3 benchmark
   - Expected keywords and required sections for generated artifacts

Usage
-----
python scripts/setup_benchmark_data.py
"""

from pathlib import Path

BENCHMARK_DIR = Path("data/benchmark")


def write_jsonl(file_path: Path, rows: list[str]):
    file_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(file_path, "w", encoding="utf-8") as file:
        for row in rows:
            file.write(row.strip() + "\n")

    print(f"Created: {file_path}")


def main():
    BENCHMARK_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    requirement_embedding_pairs = [
        '{"query":"Find requirement for account lock after failed login attempts","positive":"The system shall lock the user account after 5 failed login attempts.","negative":"The system shall allow users to export payment history as CSV."}',
        '{"query":"Create user story for password reset by email","positive":"Users shall be able to reset their password using a verified email address.","negative":"Administrators can view system audit logs."}',
        '{"query":"Generate acceptance criteria for refund within 7 days","positive":"Refunds are allowed only within 7 days after successful payment.","negative":"The system sends weekly notification emails to users."}',
        '{"query":"Create test cases for invalid login attempts","positive":"The account shall be locked after 5 consecutive failed login attempts.","negative":"The invoice download button shall be visible on the billing page."}',
        '{"query":"Draft API specification for appointment booking","positive":"The system shall allow patients to book appointments with available doctors.","negative":"Teachers can view student learning progress reports."}',
        '{"query":"Find source requirement for UAT sign-off","positive":"UAT requires acceptance criteria, test evidence, and business stakeholder sign-off.","negative":"Profile images must support PNG and JPG formats."}',

        '{"query":"Create user story for searching books by title","positive":"Users shall be able to search books using title, author name, or ISBN.","negative":"Customers can transfer funds between bank accounts."}',
        '{"query":"Generate test cases for borrowing a library book","positive":"Members shall be able to borrow available books for up to 14 days.","negative":"Insurance agents can assign leads to sales representatives."}',
        '{"query":"Create acceptance criteria for overdue book notification","positive":"The system shall send reminder notifications three days before the book due date.","negative":"Users can apply for a personal loan through mobile banking."}',
        '{"query":"Draft API specification for reserving unavailable books","positive":"Users shall be able to reserve books that are currently unavailable.","negative":"Travel agents can generate monthly commission reports."}',

        '{"query":"Create user story for transferring money between accounts","positive":"Customers shall be able to transfer funds between their own accounts.","negative":"Students can search books by author name."}',
        '{"query":"Generate acceptance criteria for beneficiary registration","positive":"Customers shall verify OTP before adding a new transfer beneficiary.","negative":"Patients can schedule appointments with doctors."}',
        '{"query":"Create test cases for failed fund transfer","positive":"The system shall display an error message when account balance is insufficient.","negative":"Library members can reserve unavailable books."}',
        '{"query":"Find requirement for loan application approval workflow","positive":"Loan applications shall require manager approval when the amount exceeds 50,000 USD.","negative":"Users shall receive reminders before book due dates."}',
        '{"query":"Draft API specification for transaction history retrieval","positive":"The system shall provide transaction history filtered by date range and account number.","negative":"Insurance agents can view policy renewal reminders."}',

        '{"query":"Create user story for assigning insurance leads","positive":"Agents shall be able to assign customer leads to sales representatives.","negative":"Customers can transfer funds between savings accounts."}',
        '{"query":"Generate acceptance criteria for policy renewal reminder","positive":"The system shall notify customers 30 days before policy expiration.","negative":"Users shall be able to search books by ISBN."}',
        '{"query":"Create test cases for insurance claim submission","positive":"Customers shall be able to upload supporting documents during claim submission.","negative":"Bank customers can register transfer beneficiaries using OTP."}',
        '{"query":"Find requirement for commission calculation","positive":"Agent commissions shall be calculated based on premium amount and policy type.","negative":"Patients shall be able to book appointments with available doctors."}',
        '{"query":"Draft API specification for customer policy lookup","positive":"Agents shall be able to retrieve policy details using policy number.","negative":"Library members can borrow books for up to 14 days."}',

        '{"query":"Generate edge case test cases for book reservation limit","positive":"A user shall not be able to reserve more than 5 books at the same time.","negative":"Customers can download bank statements as PDF."}',
        '{"query":"Create edge case test cases for transferring zero amount","positive":"The system shall reject fund transfers when the transfer amount is zero or negative.","negative":"Users can search books by ISBN."}',
        '{"query":"Generate edge cases for expired OTP during beneficiary registration","positive":"The system shall reject OTP verification if the OTP has expired.","negative":"Agents can view policy renewal reminders."}',
        '{"query":"Create edge case test cases for appointment booking outside working hours","positive":"The system shall prevent patients from booking appointments outside doctor working hours.","negative":"Loan applications require manager approval above 50,000 USD."}',

        '{"query":"Find performance requirement for book search response time","positive":"Book search results shall be returned within 2 seconds for up to 10,000 books.","negative":"Customers must verify OTP before adding a new beneficiary."}',
        '{"query":"Generate performance test cases for transaction history search","positive":"Transaction history search shall return results within 3 seconds for accounts with up to 5 years of records.","negative":"Members can borrow available books for up to 14 days."}',
        '{"query":"Create performance test cases for claim document upload","positive":"The system shall support uploading claim documents up to 20 MB within 10 seconds.","negative":"The system sends reminders three days before book due date."}',
        '{"query":"Find scalability requirement for concurrent appointment booking","positive":"The appointment booking system shall support 500 concurrent users without booking conflicts.","negative":"Agents shall retrieve policy details using policy number."}',

        '{"query":"Generate security test cases for password reset token","positive":"Password reset tokens shall expire after 15 minutes and can be used only once.","negative":"Users can reserve unavailable library books."}',
        '{"query":"Create security test cases for unauthorized transaction history access","positive":"Users shall not be able to view transaction history of accounts they do not own.","negative":"The system shall notify customers 30 days before policy expiration."}',
        '{"query":"Find security requirement for claim document access control","positive":"Only authorized agents and claim owners shall be able to view uploaded claim documents.","negative":"Book search results shall be returned within 2 seconds."}',
        '{"query":"Generate security test cases for API authentication","positive":"All API requests for customer policy lookup shall require a valid access token.","negative":"The system shall reject transfer amounts that are zero or negative."}',
    ]

    requirement_chunks = [
        '{"chunk_id":"REQ-AUTH-001","text":"The system shall lock the user account after 5 failed login attempts."}',
        '{"chunk_id":"REQ-AUTH-002","text":"Users shall be able to reset their password using a verified email address."}',
        '{"chunk_id":"REQ-PAY-001","text":"Refunds are allowed only within 7 days after successful payment."}',
        '{"chunk_id":"REQ-UAT-001","text":"UAT requires acceptance criteria, test evidence, and business stakeholder sign-off."}',
        '{"chunk_id":"REQ-BOOK-001","text":"Users shall be able to search books using title, author name, or ISBN."}',
        '{"chunk_id":"REQ-BOOK-002","text":"Members shall be able to borrow available books for up to 14 days."}',
        '{"chunk_id":"REQ-BOOK-003","text":"Users shall be able to reserve books that are currently unavailable."}',
        '{"chunk_id":"REQ-BANK-001","text":"Customers shall be able to transfer funds between their own accounts."}',
        '{"chunk_id":"REQ-BANK-002","text":"Customers shall verify OTP before adding a new transfer beneficiary."}',
        '{"chunk_id":"REQ-BANK-003","text":"The system shall display an error message when account balance is insufficient."}',
        '{"chunk_id":"REQ-BANK-004","text":"The system shall provide transaction history filtered by date range and account number."}',
        '{"chunk_id":"REQ-AGENT-001","text":"Agents shall be able to assign customer leads to sales representatives."}',
        '{"chunk_id":"REQ-AGENT-002","text":"The system shall notify customers 30 days before policy expiration."}',
        '{"chunk_id":"REQ-AGENT-003","text":"Customers shall be able to upload supporting documents during claim submission."}',
        '{"chunk_id":"REQ-AGENT-004","text":"Agents shall be able to retrieve policy details using policy number."}',
        '{"chunk_id":"REQ-EDGE-001","text":"A user shall not be able to reserve more than 5 books at the same time."}',
        '{"chunk_id":"REQ-EDGE-002","text":"The system shall reject fund transfers when the transfer amount is zero or negative."}',
        '{"chunk_id":"REQ-PERF-001","text":"Book search results shall be returned within 2 seconds for up to 10,000 books."}',
        '{"chunk_id":"REQ-PERF-002","text":"Transaction history search shall return results within 3 seconds for accounts with up to 5 years of records."}',
        '{"chunk_id":"REQ-SEC-001","text":"Password reset tokens shall expire after 15 minutes and can be used only once."}',
        '{"chunk_id":"REQ-SEC-002","text":"Users shall not be able to view transaction history of accounts they do not own."}',
    ]

    retrieval_eval = [
        '{"query":"Find requirement for account lock after failed login attempts","expected_chunk_id":"REQ-AUTH-001"}',
        '{"query":"Create user story for password reset by email","expected_chunk_id":"REQ-AUTH-002"}',
        '{"query":"Generate acceptance criteria for refund within 7 days","expected_chunk_id":"REQ-PAY-001"}',
        '{"query":"Find source requirement for UAT sign-off","expected_chunk_id":"REQ-UAT-001"}',
        '{"query":"Create user story for searching books by title","expected_chunk_id":"REQ-BOOK-001"}',
        '{"query":"Generate test cases for borrowing a library book","expected_chunk_id":"REQ-BOOK-002"}',
        '{"query":"Draft API specification for reserving unavailable books","expected_chunk_id":"REQ-BOOK-003"}',
        '{"query":"Create user story for transferring money between accounts","expected_chunk_id":"REQ-BANK-001"}',
        '{"query":"Generate acceptance criteria for beneficiary registration","expected_chunk_id":"REQ-BANK-002"}',
        '{"query":"Create test cases for failed fund transfer","expected_chunk_id":"REQ-BANK-003"}',
        '{"query":"Draft API specification for transaction history retrieval","expected_chunk_id":"REQ-BANK-004"}',
        '{"query":"Create user story for assigning insurance leads","expected_chunk_id":"REQ-AGENT-001"}',
        '{"query":"Generate acceptance criteria for policy renewal reminder","expected_chunk_id":"REQ-AGENT-002"}',
        '{"query":"Create test cases for insurance claim submission","expected_chunk_id":"REQ-AGENT-003"}',
        '{"query":"Draft API specification for customer policy lookup","expected_chunk_id":"REQ-AGENT-004"}',
        '{"query":"Generate edge case test cases for book reservation limit","expected_chunk_id":"REQ-EDGE-001"}',
        '{"query":"Create edge case test cases for transferring zero amount","expected_chunk_id":"REQ-EDGE-002"}',
        '{"query":"Find performance requirement for book search response time","expected_chunk_id":"REQ-PERF-001"}',
        '{"query":"Generate performance test cases for transaction history search","expected_chunk_id":"REQ-PERF-002"}',
        '{"query":"Generate security test cases for password reset token","expected_chunk_id":"REQ-SEC-001"}',
        '{"query":"Create security test cases for unauthorized transaction history access","expected_chunk_id":"REQ-SEC-002"}',
    ]

    llm_artifact_eval = [
        '{"task_type":"user_story","requirement":"The system shall lock the user account after 5 failed login attempts.","expected_keywords":["lock","user account","5 failed login attempts"],"required_sections":["Title","As a","I want","So that","Acceptance Criteria"]}',
        '{"task_type":"acceptance_criteria","requirement":"Refunds are allowed only within 7 days after successful payment.","expected_keywords":["refund","7 days","successful payment"],"required_sections":["Given","When","Then"]}',
        '{"task_type":"test_cases","requirement":"Users shall be able to reset their password using a verified email address.","expected_keywords":["reset password","verified email","Expected Result"],"required_sections":["Test Case ID","Scenario","Steps","Expected Result"]}',
        '{"task_type":"api_spec","requirement":"The system shall allow customers to create new orders.","expected_keywords":["create","order","customer"],"required_sections":["Endpoint","Method","Request","Response","Error"]}',
        '{"task_type":"test_cases","requirement":"The system shall reject fund transfers when the transfer amount is zero or negative.","expected_keywords":["zero","negative","reject"],"required_sections":["Test Case ID","Scenario","Steps","Expected Result"]}',
        '{"task_type":"test_cases","requirement":"Users shall not be able to view transaction history of accounts they do not own.","expected_keywords":["transaction history","accounts they do not own","unauthorized"],"required_sections":["Test Case ID","Scenario","Steps","Expected Result"]}',
    ]

    write_jsonl(
        BENCHMARK_DIR / "requirement_embedding_pairs.jsonl",
        requirement_embedding_pairs,
    )

    write_jsonl(
        BENCHMARK_DIR / "requirement_chunks.jsonl",
        requirement_chunks,
    )

    write_jsonl(
        BENCHMARK_DIR / "retrieval_eval.jsonl",
        retrieval_eval,
    )

    write_jsonl(
        BENCHMARK_DIR / "llm_artifact_eval.jsonl",
        llm_artifact_eval,
    )

    print("\nBenchmark data setup completed.")


if __name__ == "__main__":
    main()