import unittest
from fcm import Annotation


class FCMAnnotationTestCase(unittest.TestCase):
    def setUp(self):
        self.test = {'foo': 'bar'}
        self.annotation = Annotation(self.test)
    
    def test_flat_name(self):
        assert self.annotation.foo == 'bar', 'flat name lookup failed'
        assert self.annotation['foo'] == 'bar', 'index lookup failed'
        assert self.annotation.foo == \
            self.annotation['foo'], 'flat lookup isnt index lookup'
    
    def test_flat_assign(self):
        self.annotation.spam = 'eggs'
        assert self.annotation['spam'] == \
            'eggs', 'assignment lookup by index failed'
        assert self.annotation.spam == \
            'eggs', 'assignment lookup by flat failed'
        
    def test_index_assign(self):
        self.annotation['spam'] = 'eggs'
        assert self.annotation['spam'] == \
            'eggs', 'assignment lookup by index failed'
        assert self.annotation.spam == \
            'eggs', 'assignment lookup by flat failed'
    
    def test_annotation_delegation(self):
        assert self.annotation.keys()[0] == \
            'foo', 'delegation of keys() failed'
        
if __name__ == '__main__':
    suite1 = unittest.makeSuite(FCMAnnotationTestCase, 'test')

    unittest.main()
