"""
Tests for payout parsing in NetkeibaScraper.
"""
import pytest
from bs4 import BeautifulSoup
from src.scrapers.netkeiba_scraper import NetkeibaScraper


@pytest.mark.unit
class TestPayoutParsing:
    """Tests for _parse_payout_table method."""

    def setup_method(self):
        """Set up test instance."""
        self.scraper = NetkeibaScraper()

    def teardown_method(self):
        """Clean up."""
        self.scraper.close()

    def test_parse_win_payout_simple(self):
        """Test parsing a simple win payout."""
        html = """
        <table class="Payout">
            <tr>
                <th>単勝</th>
                <td>1 600円</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'win' in result
        assert len(result['win']) == 1
        assert result['win'][0]['combination'] == '1'
        assert result['win'][0]['payout'] == 600

    def test_parse_place_payout_multiple(self):
        """Test parsing multiple place payouts."""
        html = """
        <table class="Payout">
            <tr>
                <th>複勝</th>
                <td>1 150円</td>
                <td>2 300円</td>
                <td>3 200円</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'place' in result
        assert len(result['place']) == 3
        assert result['place'][0]['combination'] == '1'
        assert result['place'][0]['payout'] == 150
        assert result['place'][1]['combination'] == '2'
        assert result['place'][1]['payout'] == 300
        assert result['place'][2]['combination'] == '3'
        assert result['place'][2]['payout'] == 200

    def test_parse_quinella_payout(self):
        """Test parsing quinella (馬連) payout."""
        html = """
        <table class="Payout">
            <tr>
                <th>馬連</th>
                <td>1-2 1,500円</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'quinella' in result
        assert len(result['quinella']) == 1
        assert result['quinella'][0]['combination'] == '1-2'
        assert result['quinella'][0]['payout'] == 1500

    def test_parse_trifecta_payout(self):
        """Test parsing trifecta (3連単) payout."""
        html = """
        <table class="Payout">
            <tr>
                <th>3連単</th>
                <td>1-2-3 12,345円</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'trifecta' in result
        assert len(result['trifecta']) == 1
        assert result['trifecta'][0]['combination'] == '1-2-3'
        assert result['trifecta'][0]['payout'] == 12345

    def test_parse_payout_with_extra_whitespace(self):
        """Test parsing payout with extra whitespace and newlines."""
        html = """
        <table class="Payout">
            <tr>
                <th>単勝</th>
                <td>
                    1
                    600円
                </td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'win' in result
        assert len(result['win']) == 1
        assert result['win'][0]['combination'] == '1'
        assert result['win'][0]['payout'] == 600

    def test_parse_payout_with_yen_symbol(self):
        """Test parsing payout with ¥ symbol."""
        html = """
        <table class="Payout">
            <tr>
                <th>単勝</th>
                <td>1 ¥600</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'win' in result
        assert result['win'][0]['payout'] == 600

    def test_parse_multiple_bet_types(self):
        """Test parsing multiple bet types in one table."""
        html = """
        <table class="Payout">
            <tr>
                <th>単勝</th>
                <td>1 600円</td>
            </tr>
            <tr>
                <th>複勝</th>
                <td>1 150円</td>
                <td>2 300円</td>
            </tr>
            <tr>
                <th>馬連</th>
                <td>1-2 1,500円</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'win' in result
        assert 'place' in result
        assert 'quinella' in result
        assert len(result) == 3

    def test_parse_wide_payout_multiple(self):
        """Test parsing multiple wide (ワイド) payouts."""
        html = """
        <table class="Payout">
            <tr>
                <th>ワイド</th>
                <td>1-2 500円</td>
                <td>1-3 800円</td>
                <td>2-3 600円</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'wide' in result
        assert len(result['wide']) == 3
        assert result['wide'][0]['payout'] == 500
        assert result['wide'][1]['payout'] == 800
        assert result['wide'][2]['payout'] == 600

    def test_parse_bracket_quinella_payout(self):
        """Test parsing bracket quinella (枠連) payout."""
        html = """
        <table class="Payout">
            <tr>
                <th>枠連</th>
                <td>1-2 1,200円</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'bracket_quinella' in result
        assert result['bracket_quinella'][0]['payout'] == 1200

    def test_parse_exacta_payout(self):
        """Test parsing exacta (馬単) payout."""
        html = """
        <table class="Payout">
            <tr>
                <th>馬単</th>
                <td>1-2 2,500円</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'exacta' in result
        assert result['exacta'][0]['payout'] == 2500

    def test_parse_trio_payout(self):
        """Test parsing trio (3連複) payout."""
        html = """
        <table class="Payout">
            <tr>
                <th>3連複</th>
                <td>1-2-3 3,500円</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'trio' in result
        assert result['trio'][0]['payout'] == 3500

    def test_parse_empty_table(self):
        """Test parsing empty table."""
        html = """
        <table class="Payout">
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert result == {}

    def test_parse_invalid_payout_cell(self):
        """Test handling invalid payout cell (missing amount)."""
        html = """
        <table class="Payout">
            <tr>
                <th>単勝</th>
                <td>1</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        # Should skip invalid cells
        assert 'win' not in result or len(result.get('win', [])) == 0

    def test_parse_large_payout_amount(self):
        """Test parsing large payout amount."""
        html = """
        <table class="Payout">
            <tr>
                <th>3連単</th>
                <td>1-2-3 1,234,560円</td>
            </tr>
        </table>
        """
        soup = BeautifulSoup(html, 'lxml')
        table = soup.find('table')
        result = self.scraper._parse_payout_table(table)

        assert 'trifecta' in result
        assert result['trifecta'][0]['payout'] == 1234560
